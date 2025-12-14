# ==================== FILE: app/services/activity_search_service.py ====================

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime
import logging
import requests
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== PYDANTIC MODELS ====================

class ActivitySearchRequest(BaseModel):
    """Request model cho activity search - KHÔNG CÓ bearer_token"""
    user_role: Literal['advisor', 'student'] = Field(..., description="Role của user")
    user_id: int = Field(..., description="ID của user")
    from_date: Optional[str] = Field(None, description="Lọc từ ngày")
    to_date: Optional[str] = Field(None, description="Lọc đến ngày")
    status: Optional[Literal['upcoming', 'completed', 'cancelled']] = Field(None, description="Trạng thái")
    title: Optional[str] = Field(None, description="Tên hoạt động")
    point_type: Optional[Literal['ctxh', 'ren_luyen']] = Field(None, description="Loại điểm")
    organizer_unit: Optional[str] = Field(None, description="Tên đơn vị")


class ActivityRole(BaseModel):
    activity_role_id: int
    role_name: str
    description: Optional[str] = None
    requirements: Optional[str] = None
    points_awarded: int
    point_type: Literal['ctxh', 'ren_luyen']
    max_slots: Optional[int] = None


class ActivityClass(BaseModel):
    class_id: int
    class_name: str


class ActivityUnit(BaseModel):
    unit_id: int
    unit_name: str


class ActivityAdvisor(BaseModel):
    advisor_id: int
    full_name: str


class Activity(BaseModel):
    activity_id: int
    title: str
    general_description: Optional[str] = None
    location: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    status: str
    advisor: Optional[ActivityAdvisor] = None
    organizer_unit: Optional[ActivityUnit] = None
    classes: List[ActivityClass] = []
    roles: List[ActivityRole] = []


class ActivitySearchResponse(BaseModel):
    success: bool
    data: List[Activity]
    total: int = Field(default=0)
    search_type: str = Field(default="activity_search")
    source: str = Field(default="activity")
    activities_raw: Optional[List[dict]] = Field(None)
    error_message: Optional[str] = Field(None)


class ActivitySearchWithLLMResponse(BaseModel):
    success: bool
    llm_response: str
    activities: List[Activity]
    activities_raw: Optional[List[dict]] = Field(None)
    total: int
    search_type: str = Field(default="activity_search_with_llm")
    source: str = Field(default="activity")
    error_message: Optional[str] = Field(None)


# ==================== ACTIVITY SEARCH SERVICE ====================

class ActivitySearchService:
    def __init__(self):
        self.api_base_url = os.getenv('LARAVEL_API_URL', 'http://localhost:8000/api')
        self.api_key = os.getenv('GOOGLE_API_KEY')
        
        if self.api_key:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=self.api_key,
                temperature=0.3,
            )
        else:
            self.llm = None

    def search_activities(self, request: ActivitySearchRequest, bearer_token: str) -> ActivitySearchResponse:
        """Tìm kiếm hoạt động - bearer_token là REQUIRED parameter"""
        try:
            if not bearer_token:
                logger.error("[API] Token không được cung cấp")
                return ActivitySearchResponse(
                    success=False, 
                    data=[], 
                    total=0,
                    error_message="Token xác thực không được cung cấp"
                )
            
            logger.info(f"[API] Gọi API với token: {bearer_token[:40]}...")
            
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Authorization': f'Bearer {bearer_token}'
            }
            
            params = {}
            if request.from_date:
                params['from_date'] = request.from_date
            if request.to_date:
                params['to_date'] = request.to_date
            if request.status:
                params['status'] = request.status
            if request.title:
                params['title'] = request.title
            if request.point_type:
                params['point_type'] = request.point_type
            if request.organizer_unit:
                params['organizer_unit'] = request.organizer_unit
            
            logger.info(f"[API] Gọi {self.api_base_url}/activities với params: {params}")
            
            response = requests.get(
                f"{self.api_base_url}/activities",
                headers=headers,
                params=params,
                timeout=30
            )
            
            logger.info(f"[API] Response status: {response.status_code}")
            
            if response.status_code == 401:
                error_detail = response.text[:200]
                logger.error(f"[API] 401 - Token không hợp lệ. Response: {error_detail}")
                return ActivitySearchResponse(
                    success=False, data=[], total=0,
                    error_message="Token không hợp lệ hoặc đã hết hạn"
                )
            
            if response.status_code == 403:
                return ActivitySearchResponse(
                    success=False, data=[], total=0,
                    error_message="Không có quyền truy cập"
                )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    activities_data = data.get('data', [])
                    
                    activities = []
                    for act_data in activities_data:
                        try:
                            activity = Activity(**act_data)
                            activities.append(activity)
                        except Exception as e:
                            logger.warning(f"Lỗi parse activity: {e}")
                            continue
                    
                    logger.info(f"[API] Lấy được {len(activities)} hoạt động")
                    
                    return ActivitySearchResponse(
                        success=True,
                        data=activities,
                        total=len(activities),
                        activities_raw=activities_data
                    )
                else:
                    error_msg = data.get('message', 'Lỗi không xác định')
                    return ActivitySearchResponse(
                        success=False, data=[], total=0,
                        error_message=error_msg
                    )
            else:
                return ActivitySearchResponse(
                    success=False, data=[], total=0,
                    error_message=f'HTTP {response.status_code}: {response.text[:100]}'
                )
                
        except requests.exceptions.Timeout:
            return ActivitySearchResponse(
                success=False, data=[], total=0,
                error_message="Request timeout"
            )
        except Exception as e:
            logger.error(f"[API] Lỗi: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return ActivitySearchResponse(
                success=False, data=[], total=0,
                error_message="Lỗi hệ thống"
            )

    def search_with_llm(self, request: ActivitySearchRequest, bearer_token: str) -> ActivitySearchWithLLMResponse:
        """Tìm kiếm với LLM"""
        try:
            search_result = self.search_activities(request, bearer_token)
            
            if not search_result.success:
                return ActivitySearchWithLLMResponse(
                    success=False,
                    llm_response=search_result.error_message or "Không thể tìm kiếm",
                    activities=[], 
                    activities_raw=[], 
                    total=0,
                    source="activity", 
                    error_message=search_result.error_message
                )
            
            if not search_result.data:
                return ActivitySearchWithLLMResponse(
                    success=True,
                    llm_response="Hiện tại không có hoạt động nào phù hợp với yêu cầu của bạn.",
                    activities=[], 
                    activities_raw=[], 
                    total=0,
                    source="activity"
                )
            
            llm_response = "Không thể tạo tóm tắt do lỗi LLM."
            
            if self.llm and search_result.data:
                try:
                    context_parts = []
                    for i, activity in enumerate(search_result.data[:10], 1):
                        roles_info = []
                        for role in activity.roles:
                            roles_info.append(
                                f"  - {role.role_name}: {role.points_awarded} điểm ({role.point_type})"
                                + (f", Số chỗ: {role.max_slots}" if role.max_slots else "")
                            )
                        
                        activity_text = f"""
**Hoạt động {i}: {activity.title}**
- Thời gian: {activity.start_time or 'Chưa xác định'} → {activity.end_time or 'Chưa xác định'}
- Địa điểm: {activity.location or 'Chưa xác định'}
- Trạng thái: {activity.status}
- Đơn vị tổ chức: {activity.organizer_unit.unit_name if activity.organizer_unit else 'Không rõ'}
- Mô tả: {activity.general_description or 'Không có'}
- Các vai trò tham gia:
{chr(10).join(roles_info) if roles_info else '  Không có vai trò nào'}
"""
                        context_parts.append(activity_text)
                    
                    context = "\n".join(context_parts)
                    
                    prompt_template = PromptTemplate(
                        input_variables=["context", "total", "filters"],
                        template="""
Vai trò:
Bạn là trợ lý AI chuyên tổng hợp thông tin về các hoạt động sinh viên.

Nhiệm vụ:
Hãy tóm tắt các hoạt động sau đây một cách rõ ràng, dễ hiểu cho sinh viên và cố vấn.

Tiêu chí tìm kiếm:
{filters}

Kết quả: Tìm thấy {total} hoạt động

Danh sách hoạt động:
{context}

Yêu cầu trả lời:
1. **Tóm tắt tổng quan** (1-2 câu về số lượng và loại hoạt động)
2. **Danh sách hoạt động nổi bật** (liệt kê 3-5 hoạt động quan trọng nhất):
   - Tên hoạt động
   - Thời gian và địa điểm
   - Điểm rèn luyện/CTXH có thể đạt được
   - Vai trò có thể tham gia
3. **Hướng dẫn đăng ký** (nếu có hoạt động sắp diễn ra):
   - Lưu ý thời gian đăng ký
   - Số lượng chỗ còn trống (nếu có)
4. **Gợi ý** (nếu có nhiều hơn 5 hoạt động, khuyến nghị user lọc thêm)

**LƯU Ý QUAN TRỌNG:**
- Bên dưới phần tóm tắt này, hệ thống sẽ hiển thị danh sách chi tiết các hoạt động
- Sinh viên có thể click vào từng hoạt động để xem chi tiết và đăng ký
- Do đó, trong phần tóm tắt của bạn, hãy nhắc nhở: "Chi tiết và đăng ký hoạt động ở danh sách bên dưới"

Sử dụng markdown để format rõ ràng với **bold** cho tiêu đề và bullet points cho danh sách.
"""
                    )
                    
                    filters_summary = []
                    if request.from_date:
                        filters_summary.append(f"Từ ngày: {request.from_date}")
                    if request.to_date:
                        filters_summary.append(f"Đến ngày: {request.to_date}")
                    if request.status:
                        filters_summary.append(f"Trạng thái: {request.status}")
                    if request.title:
                        filters_summary.append(f"Tên hoạt động: {request.title}")
                    if request.point_type:
                        filters_summary.append(f"Loại điểm: {request.point_type}")
                    if request.organizer_unit:
                        filters_summary.append(f"Đơn vị: {request.organizer_unit}")
                    
                    filters_text = "\n".join(filters_summary) if filters_summary else "Không có bộ lọc cụ thể"
                    
                    prompt = prompt_template.format(
                        context=context,
                        total=search_result.total,
                        filters=filters_text
                    )
                    
                    llm_response = self.llm.invoke(prompt).content
                    logger.info(f"[LLM] Đã tạo response cho {search_result.total} hoạt động")
                    
                except Exception as e:
                    logger.error(f"[LLM] Lỗi: {e}")
                    llm_response = "Không thể tạo tóm tắt do lỗi hệ thống."
            
            return ActivitySearchWithLLMResponse(
                success=True,
                llm_response=llm_response,
                activities=search_result.data,
                activities_raw=search_result.activities_raw,
                total=search_result.total,
                source="activity"
            )
            
        except Exception as e:
            logger.error(f"[LLM] Lỗi: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return ActivitySearchWithLLMResponse(
                success=False,
                llm_response="Lỗi hệ thống khi tìm kiếm hoạt động",
                activities=[], 
                activities_raw=[], 
                total=0,
                source="activity"
            )


# ==================== GLOBAL SERVICE INSTANCE ====================

activity_search_service = ActivitySearchService()


# ==================== WRAPPER FUNCTIONS WITH AUTO-INJECTED TOKEN ====================

# BIẾN GLOBAL để lưu token - sẽ được set từ bên ngoài
_current_bearer_token: Optional[str] = None

def set_bearer_token(token: str):
    """Set token globally để wrapper có thể dùng"""
    global _current_bearer_token
    _current_bearer_token = token
    logger.info(f"[TOKEN] Bearer token đã được set: {token[:40] if token else 'None'}...")


def activity_search_wrapper(
    user_role, 
    user_id, 
    from_date=None, 
    to_date=None, 
    status=None, 
    title=None, 
    point_type=None, 
    organizer_unit=None
):
    """
    Wrapper TỰ ĐỘNG INJECT TOKEN từ global variable
    LLM KHÔNG CẦN truyền bearer_token nữa!
    """
    global _current_bearer_token
    
    if not _current_bearer_token:
        logger.error("[WRAPPER] Token chưa được set!")
        return {
            "success": False,
            "total": 0,
            "source": "activity",
            "activities_raw": [],
            "summary": "Lỗi xác thực: Token không được cung cấp"
        }
    
    logger.info(f"[WRAPPER] activity_search với auto-injected token")
    
    request = ActivitySearchRequest(
        user_role=user_role,
        user_id=user_id,
        from_date=from_date,
        to_date=to_date,
        status=status,
        title=title,
        point_type=point_type,
        organizer_unit=organizer_unit
    )
    
    # TỰ ĐỘNG INJECT TOKEN
    result = activity_search_service.search_activities(request, bearer_token=_current_bearer_token)
    
    output = {
        "success": result.success,
        "total": result.total,
        "source": "activity",
        "activities_raw": result.activities_raw if result.activities_raw else [],
        "summary": f"Tìm thấy {result.total} hoạt động" if result.success else (result.error_message or "Lỗi tìm kiếm")
    }
    
    logger.info(f"[WRAPPER] Trả về {result.total} hoạt động")
    return output


def activity_search_with_llm_wrapper(
    user_role, 
    user_id, 
    from_date=None, 
    to_date=None, 
    status=None, 
    title=None, 
    point_type=None, 
    organizer_unit=None
):
    """
    Wrapper TỰ ĐỘNG INJECT TOKEN
    """
    global _current_bearer_token
    
    if not _current_bearer_token:
        logger.error("[WRAPPER] Token chưa được set!")
        return {
            "success": False,
            "llm_response": "Lỗi xác thực: Token không được cung cấp",
            "total": 0,
            "source": "activity",
            "activities_raw": [],
            "error_message": "Token không được cung cấp"
        }
    
    logger.info(f"[WRAPPER] activity_search_with_llm với auto-injected token")
    
    request = ActivitySearchRequest(
        user_role=user_role,
        user_id=user_id,
        from_date=from_date,
        to_date=to_date,
        status=status,
        title=title,
        point_type=point_type,
        organizer_unit=organizer_unit
    )
    
    # TỰ ĐỘNG INJECT TOKEN
    result = activity_search_service.search_with_llm(request, bearer_token=_current_bearer_token)
    
    output = {
        "success": result.success,
        "llm_response": result.llm_response,
        "total": result.total,
        "source": "activity",
        "activities_raw": result.activities_raw if result.activities_raw else [],
        "error_message": result.error_message
    }
    
    logger.info(f"[WRAPPER] Trả về LLM response với {result.total} hoạt động")
    return output


# ==================== LANGCHAIN TOOLS (CẬP NHẬT DESCRIPTION) ====================

activity_search_tool = StructuredTool.from_function(
    func=activity_search_wrapper,
    name="activity_search",
    description="""
Tìm kiếm hoạt động ngoại khóa.

QUAN TRỌNG: Tool này TỰ ĐỘNG sử dụng token xác thực của user hiện tại.
Bạn KHÔNG CẦN và KHÔNG ĐƯỢC truyền bearer_token.

Parameters:
- user_role (str, required): "advisor" hoặc "student"
- user_id (int, required): ID của user
- from_date (str, optional): Lọc từ ngày (format: YYYY-MM-DD)
- to_date (str, optional): Lọc đến ngày (format: YYYY-MM-DD)
- status (str, optional): "upcoming", "completed", "cancelled"
- title (str, optional): Tên hoạt động
- point_type (str, optional): "ctxh" hoặc "ren_luyen"
- organizer_unit (str, optional): Tên đơn vị

Returns: Dict với activities_raw và metadata

Example usage:
activity_search(user_role="student", user_id=123, status="upcoming")
"""
)

activity_search_with_llm_tool = StructuredTool.from_function(
    func=activity_search_with_llm_wrapper,
    name="activity_search_with_summary",
    description="""
Tìm kiếm hoạt động và tạo tóm tắt bằng LLM.

QUAN TRỌNG: Tool này TỰ ĐỘNG sử dụng token xác thực của user hiện tại.
Bạn KHÔNG CẦN và KHÔNG ĐƯỢC truyền bearer_token.

Parameters: Giống activity_search

Returns: Dict với llm_response, activities_raw và metadata

Example usage:
activity_search_with_summary(user_role="student", user_id=123, from_date="2025-03-01")
"""
)


def get_activity_search_tools():
    """Get all activity search tools for LangGraph"""
    return [activity_search_tool, activity_search_with_llm_tool]