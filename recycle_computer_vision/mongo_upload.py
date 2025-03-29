import os
import mongoengine as me

# Connect to MongoDB (update parameters if needed)
me.connect(
    db="RecycleImage",
    host="mongodb+srv://Lyrics:lyrikal1216S!@gosustainyourself.tsdvv.mongodb.net/GoSustainYourself"
)

from mongoengine import Document, FileField, StringField

class RecycleImage(Document):
    filename = StringField(required=True)
    image_file = FileField(required=True)

def upload_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'rb') as f:
                        data = f.read()
                    image_doc = RecycleImage(filename=file)
                    image_doc.image_file.put(data, content_type='image/jpeg')
                    image_doc.save()
                    print(f"Uploaded: {file_path}")
                except Exception as e:
                    print(f"Error uploading {file_path}: {e}")

if __name__ == '__main__':
    # Set the directory where your image files are stored.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_directory = os.path.join(script_dir, 'data', 'Main Dataset')
    upload_images(image_directory)