import asyncio
from fastapi import APIRouter, HTTPException, Request

from app.schemas import ProcessVideoRequest, ProcessVideoResponse
from app.services import pipeline_service

router = APIRouter()


@router.post("/process-video", response_model=ProcessVideoResponse)
async def process_video(req: ProcessVideoRequest, request: Request) -> ProcessVideoResponse:
    detector  = request.app.state.detector
    extractor = request.app.state.extractor

    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            pipeline_service.process_video,
            req, detector, extractor,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    request.app.state.stream_manager.update_identities(result.identities)
    return result
