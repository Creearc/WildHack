from exif import Image
import os

path = 'dataset/p2'

for f in os.listdir(path):
  with open('{}/{}'.format(path, f), "rb") as file:
    image = Image(file)
    print(dir(image))
    print('Orientation: ', image.orientation)
    print(image.subject_distance)
    print(f"Latitude: {image.gps_latitude} {image.gps_latitude_ref}")
    print(f"Longitude: {image.gps_longitude} {image.gps_longitude_ref}\n")
    print('_________________')
