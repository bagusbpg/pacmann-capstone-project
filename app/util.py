import Levenshtein

def response(code, message):
    return {
        'code': code,
        'message': message
    }

def most_similar():
    