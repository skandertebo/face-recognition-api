from pydantic import *

class AnalyseImageDto(BaseModel):
  image_url: str
  previous_face_encodings: dict
  