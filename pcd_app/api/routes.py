from fastapi import APIRouter, Depends, HTTPException

router = APIRouter()

ul_usecase = injector.get(UploadUsecase)  # type: ignore
dl_usecase = injector.get(DownloadUsecase)  # type: ignore


@router.get(
    "/a/{a}/b/{b}/c/",
)
async def download_pcd_from_site():
    pass
