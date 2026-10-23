import os
import numpy as np
from PIL import Image
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from datetime import datetime


def png_to_dicom(png_path, output_path, patient_id, study_id):
    img = Image.open(png_path).convert('L')
    arr = np.array(img)
    
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    
    ds = Dataset()
    ds.file_meta = file_meta
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.PatientID = patient_id
    ds.StudyInstanceUID = study_id
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyDate = datetime.now().strftime('%Y%m%d')
    ds.SeriesDate = datetime.now().strftime('%Y%m%d')
    ds.Modality = 'CR'
    ds.Rows = arr.shape[0]
    ds.Columns = arr.shape[1]
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.PixelData = arr.tobytes()
    
    ds.save_as(output_path, write_like_original=False)


def convert_batch(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for img_file in os.listdir(input_dir):
        if img_file.endswith('.png'):
            png_path = os.path.join(input_dir, img_file)
            patient_id = img_file.split('_')[1].replace('.png', '')
            study_id = pydicom.uid.generate_uid()
            output_path = os.path.join(output_dir, f'{patient_id}.dcm')
            png_to_dicom(png_path, output_path, patient_id, study_id)
