ALLOWED_EXTENSIONS = {'csv', 'json', 'xlsx'}
def generate_uuid():
	return str(uuid.uuid4())

# check if the file is part of the allowed extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS