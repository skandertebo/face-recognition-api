import json
import os
import face_recognition
from dto.analyse_image_dto import AnalyseImageDto
from PIL import Image
import requests
from io import *
import face_recognition
import numpy as np

def load_image_from_url(url):
    """
    Load an image from a URL and convert it to a file object.

    Parameters:
    url (str): The URL of the image to load.

    Returns:
    Image: The PIL Image object loaded from the URL.
    """
    response = requests.get(url)
    print(response.status_code)
    image = Image.open(BytesIO(response.content))
    return image

def convert_image_to_file_object(image):
    """
    Convert a PIL Image object to a file-like object.

    Parameters:
    image (Image): A PIL Image object.

    Returns:
    io.BytesIO: A file-like object containing the image data.
    """
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr





def get_faces_from_image(analyse_image_dto: AnalyseImageDto):
    url = analyse_image_dto.image_url
    previous_encodings = analyse_image_dto.previous_face_encodings
    image_from_url_response = load_image_from_url(url)
    image_file = convert_image_to_file_object(image_from_url_response)
    image = face_recognition.load_image_file(image_file)
    face_locations = face_recognition.face_locations(image)
    print('found {} faces in image'.format(len(face_locations)))
    face_encodings = face_recognition.face_encodings(image)
    result = {
        "faces":[]
    }
    for face_loc, index in zip(face_locations, range(len(face_locations))):
        print(f"Face {index} is located at pixel location Top: {face_loc[0]}, Right: {face_loc[1]}, Bottom: {face_loc[2]}, Left: {face_loc[3]}")
        face = dict()
        bbox_top_in_percentage = face_loc[0] / image_from_url_response.height
        bbox_left_in_percentage = face_loc[3] / image_from_url_response.width
        bbox_width_in_percentage = (face_loc[1] - face_loc[3]) / image_from_url_response.width
        bbox_height_in_percentage = (face_loc[2] - face_loc[0]) / image_from_url_response.height
        face["bbox_top"] = bbox_top_in_percentage
        face["bbox_left"] = bbox_left_in_percentage
        face["bbox_width"] = bbox_width_in_percentage
        face["bbox_height"] = bbox_height_in_percentage
        for face_group_id, face_group_encoding in previous_encodings.items():
            print(f"Comparing face {index} to face group {face_group_id}")
            compare_result = face_recognition.compare_faces([np.array(face_group_encoding)], face_encodings[index])
            print(f"Face {index} is a match for face group {face_group_id}: {compare_result[0]}")
            if compare_result[0] == True:
                face["face_group_id"] = face_group_id
                break
        else:
            print(f"Face {index} is a new face")
            face["face_group_id"] = None
            face["face_encoding"] = face_encodings[index].tolist()
        
        result["faces"].append(face)
    return result   


""" dto = AnalyseImageDto(image_url="https://newcelebratelife.netlify.app/_next/image?url=https%3A%2F%2Fcelebr8-v2-files.s3.amazonaws.com%2Fdev%2Fmemories%2F6554d0c4dfef9ca0c94cc9c8%2F65758ae8e8a499bc60d938af-t%25C3%2583%25C2%25A9l%25C3%2583%25C2%25A9chargement%2520%25286%2529.jpeg&w=1920&q=75", previous_face_encodings={})
result = get_faces_from_image(dto)
face_encoding = result["faces"][0]["face_encoding"]
print(face_encoding)
#get_current_directory
current_directory = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_directory,'example_encodings.json'), mode='w') as file:
    print([face_encoding.tolist()])
    # file has json array of face encodings
    # append json array of face encodings
    file.write(json.dumps([face_encoding.tolist()])) """





