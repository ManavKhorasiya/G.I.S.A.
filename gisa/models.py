from django.db import models
from django_resized import ResizedImageField
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from project.settings import BASE_DIR, MEDIA_ROOT
import os
 
# fs = FileSystemStorage(location='/media/segmented_images/')

segmented_superpath = os.path.join(BASE_DIR, 'media/')

class OverwriteStorage(FileSystemStorage):
    #overwriting image files with same name
    def get_available_name(self, name, max_length = None):
        if self.exists(name):
            os.remove(os.path.join(settings.MEDIA_ROOT, name))
        return name


class Image(models.Model) :
    IMAGE_CHOICES = [
        ('Number', 'Sign Language Number'),
        ('Alphabet', 'Sign Language Alphabet')
    ]
    category = models.CharField(max_length = 256,default = 'Sign Language Hand Gesture Digit',choices = IMAGE_CHOICES)
    uploads = ResizedImageField(size = [200,200], upload_to = 'uploaded_images/' , storage = OverwriteStorage())
    
    def __str__(self):
        return str(self.category)

class segment(models.Model):
    id = models.AutoField(primary_key = True)
    IMAGE_CHOICES = [
        ('Number', 'Sign Language Number'),
        ('Alphabet', 'Sign Language Alphabet')
    ]
    category = models.CharField(max_length = 256,default = 'Sign Language Hand Gesture Digit',choices = IMAGE_CHOICES)
    uploads = ResizedImageField(size = [200,200],upload_to = 'segmented_images/', blank = True) #, storage=OverwriteStorage())