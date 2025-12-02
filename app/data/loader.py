from app.data.loader import load_places, to_documents
places = load_places()
docs = to_documents(places)
