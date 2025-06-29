import logging

from fastapi import APIRouter, status, UploadFile, File, HTTPException
from starlette.responses import JSONResponse

from core.config import settings
from services.file_service import FileService

router = APIRouter(prefix="/api/files")

file_service = FileService()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@router.post("", status_code=201)
def upload_file(file: UploadFile = File(...)):
    try:
        if not file:
            raise ValueError("No file provided")

        file.file.seek(0, 2)  # Moves the internal pointer of the file to the end
        file_size = file.file.tell()
        file.file.seek(0)

        if file_size == 0:
            return ValueError("File is empty!")

        if file_size > 50 * 1024 * 1024:
            raise ValueError("File too large (max 50MB)")

        logger.info(f"Uploading {file.filename}, file_size: {file_size} bytes.")

        file_service.store_to_vectorstore(
            file=file,
            session_id=settings.session_id,
            user_id=settings.user_id
        )

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={"message": "File uploaded and stored successfully!"}
        )
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except RuntimeError as re:
        logger.error(f"Runtime error: {str(re)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(re)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing error: {str(e)}"
        )
