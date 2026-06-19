
import unicodedata
import re
STOPWORDS = {
    # Artículos
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    # Preposiciones
    "a", "ante", "bajo", "con", "de", "desde", "en", "entre",
    "hacia", "hasta", "para", "por", "sin", "sobre", "tras",
    # Conjunciones
    "pero", "sino", "que", "aunque", "si",
    # Pronombres
    "yo", "tu", "el", "ella", "nosotros", "vosotros", "ellos", "ellas",
    "me", "te", "se", "nos", "lo", "le", "les",
    # Verbos comunes
    "es", "son", "fue", "ser", "estar", "hay", "tiene", "tienen",
    "ha", "han", "era", "sido", "esto", "esta", "eso",
    # Otros
    "como", "mas", "su", "sus", "al", "del", "tambien",
    "si", "no", "ya", "muy", "bien", "cuando", "donde", "quien"
}

def normalize(text:str) -> str:
    text=text.lower()
    te=unicodedata.normalize('NFKD', text) #Separa letra de acentos
    #Letra= "Ll"
    #Acento= "Mn"
    result="" 
    for letra in te:
        categoria=unicodedata.category(letra)
        if categoria != 'Mn': #Si es distinto a un acento se agrega al resultado
            result+=letra
    return result
def tokenize(text:str) -> list[str]:
    letras = [token for token in re.findall(r'\w+', text) if len(token) > 1]
    return letras

def remove_stopwords(tokens,stopwords=STOPWORDS) -> list[str]:
    removed =list(filter(lambda token: token not in stopwords, tokens))
    return removed

def preprocess(text:str,stopword=STOPWORDS) -> list[str]:
    return remove_stopwords(tokenize(normalize(text)),stopword)
