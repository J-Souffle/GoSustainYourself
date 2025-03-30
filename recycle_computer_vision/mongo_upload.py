import os
import mongoengine as me

def upload_images_by_type(directory):
    # The folder name is used as the MongoDB database name (e.g. "batteries").
    item_type = os.path.basename(directory)  # "batteries", etc.
    me.connect(db=item_type, host="mongodb://atlas-sql-67e83187da002a438096b0e9-tsdvv.a.query.mongodb.net/RecycledDataset?ssl=true&authSource=admin")

    from mongoengine import Document, FileField, StringField

    class RecycleImage(Document):
        filename = StringField(required=True)
        image_file = FileField(required=True)

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
                    print(f"Uploaded to DB '{item_type}': {file_path}")
                except Exception as e:
                    print(f"Error uploading {file_path}: {e}")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Example subdirectory named "batteries", storing images in a DB also called "batteries"
    batteries_directory = os.path.join(script_dir, 'data', 'batteries')
    upload_images_by_type(batteries_directory)

    # You could repeat for other directories (e.g., "paper", "plastics") if needed.