import numpy as np

def prepare_image(file):
    if not file:
        return None, response(400, 'no image uploaded')
    
    if file.content_type != 'image/jpeg':
        return None, response(415, 'unallowed image type')
    
    real_file_size = 0
    temp: IO = NamedTemporaryFile(delete=False)
    for chunk in file.file:
        real_file_size += len(chunk)
        if real_file_size > LIMIT:
            return None, response(413, 'image size is too large')
        try:
            temp.write(chunk)
        except:
            return None, response(500, 'failed to save image')
    temp.close()
    
    imagePath = f'./{time.time_ns()}.jpg'
    os.rename(temp.name, imagePath)

    return imagePath, None

def most_similar(checkedInCars, text, threshold=0.85):
    maxSimilarCount = 0
    mostSimilarText = ''
    mostSimilarId = ''
    for value in checkedInCars:
        id, textToCompare, _ = value
        shortText, longText = text, textToCompare
        if shortText > longText:
            shortText, longText = longText, shortText
        
        similarCount = 0
        for idx in range(len(shortText)):
            if shortText[idx] != longText[idx]:
                if similarCount > maxSimilarCount:
                    maxSimilarCount = similarCount
                    mostSimilarText = textToCompare
                    mostSimilarId = id
                break
            similarCount += 1
            
        if similarCount > maxSimilarCount:
            maxSimilarCount = similarCount
            mostSimilarText = textToCompare
            mostSimilarId = id

    ratioOfSimilarity = maxSimilarCount/len(text)
    if ratioOfSimilarity >= threshold:
        return mostSimilarId, None
    
    return None, response(400, 'no matching license number is found')

def response(code, message):
    return {
        'code': code,
        'message': message
    }
