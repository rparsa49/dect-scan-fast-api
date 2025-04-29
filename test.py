import pydicom

def get_dicom_kvp(dicom_path):
    """Reads a DICOM file and returns its KVP value."""
    try:
        dicom_data = pydicom.dcmread(dicom_path)
        kvp = dicom_data.get("KVP", None)
        if kvp is not None:
            return kvp
        else:
            raise ValueError("KVP value not found in the DICOM file.")
    except Exception as e:
        raise RuntimeError(f"Error reading DICOM file: {e}")


# Example usage
path = "/Users/royaparsa/Downloads/test-data/1.3.12.2.1107.5.1.4.83775.30000024051312040257200020230/CT1.3.12.2.1107.5.1.4.83775.30000024051312040257200020231.dcm"
kvp_value = get_dicom_kvp(path)
print(f"KVP Value: {kvp_value}")
