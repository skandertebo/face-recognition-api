from fastapi import FastAPI
from dto.analyse_image_dto import AnalyseImageDto
from faces_service import *
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
  
  
@app.post("/analyse")
def analyse(analysis_dto: AnalyseImageDto):
  result = get_faces_from_image(analysis_dto)
  return result

  