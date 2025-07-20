import praw
from textblob import TextBlob
from datetime import datetime, timezone


from libs.reddit_config import (
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USERNAME,
    REDDIT_PASSWORD,
    REDDIT_USER_AGENT
)


def connect_reddit():

    try: 
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            username=REDDIT_USERNAME,
            password=REDDIT_PASSWORD,
            user_agent=REDDIT_USER_AGENT
        )
        
        # Verifica autenticación
        #print("Autenticando...")
        #user = reddit.user.me()
        #print(f"✅ Autenticación exitosa: {user}")
       
        return reddit
    except Exception as e:
        print(f"❌ Error de autenticación: {e}")

def obtener_menciones_en_reddit(simbolo, limite=50):

    try: 
        reddit = connect_reddit()
    
        subreddits = ["stocks", "wallstreetbets", "options", "investing"]
        menciones = []
    
        for sub in subreddits:
            subreddit = reddit.subreddit(sub)
            for post in subreddit.search(simbolo, sort="new", limit=limite):
                menciones.append({
                    "subreddit": sub,
                    "titulo": post.title,
                    "fecha": post.created_utc,
                    "link": f"https://reddit.com{post.permalink}",
                    "upvotes": post.score
                })
    
        return menciones
    except Exception as e:
        print(f"❌ Error de autenticación: {e}")



def obtener_posts_reddit(simbolos, limite=30):
    
    reddit = connect_reddit()

    subreddits = ["stocks", "wallstreetbets", "options", "investing"]
    posts_relevantes = []

    for subreddit in subreddits:
        for post in reddit.subreddit(subreddit).hot(limit=limite):
            for simbolo in simbolos:
                if simbolo.lower() in post.title.lower():
                    sentimiento = TextBlob(post.title).sentiment.polarity
                    clasificacion = "Positivo" if sentimiento > 0.1 else "Negativo" if sentimiento < -0.1 else "Neutral"

                    posts_relevantes.append({
                        "subreddit": subreddit,
                        "titulo": post.title,
                        "score": post.score,
                        "comentarios": post.num_comments,
                        "fecha": datetime.fromtimestamp(post.created_utc, tz=timezone.utc).strftime('%Y-%m-%d %H:%M'),
                        "url": post.url,
                        "sentimiento": clasificacion
                    })
                    break  # Evitar duplicados si un post menciona más de un símbolo

    return posts_relevantes



def obtener_data_sentimientos_reddit(simbolos, limite=30):
    reddit = connect_reddit()

    # Este diccionario se debe construir dinámicamente en tu lógica de scraping Reddit
    #sentimientos_por_simbolo = {
    #    "AAPL": {"Positivo": 12, "Neutral": 5, "Negativo": 3},
    #    "TSLA": {"Positivo": 18, "Neutral": 7, "Negativo": 4},
    #    "META": {"Positivo": 7, "Neutral": 3, "Negativo": 6}
    #}


     # Crear lista para almacenar datos
    sentimientos = []
    
    # Procesar publicaciones de Reddit
    for post in reddit.subreddit("wallstreetbets").search(simbolos, limit=100):
        fecha = datetime.fromtimestamp(post.created_utc).date()
        texto = post.title + " " + post.selftext
        analisis = TextBlob(texto)
        sentimiento = "Positivo" if analisis.sentiment.polarity > 0.1 else "Negativo" if analisis.sentiment.polarity < -0.1 else "Neutral"
    
        sentimientos.append({
            "Fecha": fecha,
            "Símbolo": simbolos.upper(),
            "Sentimiento": sentimiento,
            "Cantidad": 1
        })

    return sentimientos

