import random
import pandas as pd

random.seed(42)
NUM_TICKETS = 500

TICKETS = {
    "Amministrazione": [
        {
            "title": "Cambio intestazione fattura",
            "body": [
                "Chiedo il cambio di intestazione della fattura a nome dell’associazione",
                "La fattura deve essere intestata a un ente diverso",
                "È possibile modificare l’intestatario della fattura emessa?",
                "Serve correggere l’intestazione della fattura già ricevuta"
            ],
            "priority": {"bassa": 0.4, "media": 0.5, "alta": 0.1}
        },
        {
            "title": "Errore importo fattura",
            "body": [
                "L’importo indicato in fattura non corrisponde a quanto pagato",
                "La cifra riportata in fattura è errata",
                "Ho notato un errore nell’importo della fattura ricevuta"
            ],
            "priority": {"bassa": 0.2, "media": 0.5, "alta": 0.3}
        },
        {
            "title": "Pagamento non registrato",
            "body": [
                "Ho effettuato il pagamento ma non risulta registrato",
                "Il pagamento è stato effettuato ma lo stato risulta ancora non pagato",
                "Ho pagato la fattura ma il sistema non lo segnala"
            ],
            "priority": {"bassa": 0.1, "media": 0.4, "alta": 0.5}
        },
        {
            "title": "Richiesta rimborso",
            "body": [
                "Chiedo il rimborso per una visita annullata",
                "La visita è stata cancellata, come posso ottenere il rimborso?",
                "Vorrei richiedere il rimborso della prenotazione"
            ],
            "priority": {"bassa": 0.2, "media": 0.4, "alta": 0.4}
        }
    ],

    "Tecnico": [
        {
            "title": "Errore prenotazione visita",
            "body": [
                "Durante la prenotazione della visita il sistema va in errore",
                "Non riesco a completare la prenotazione online",
                "La procedura di prenotazione si blocca"
            ],
            "priority": {"bassa": 0.1, "media": 0.4, "alta": 0.5}
        },
        {
            "title": "Modulo contatti non funziona",
            "body": [
                "Il modulo di contatto non permette l’invio del messaggio",
                "Il form contatti restituisce un errore",
                "Non riesco a inviare una richiesta tramite il modulo"
            ],
            "priority": {"bassa": 0.2, "media": 0.5, "alta": 0.3}
        },
        {
            "title": "Sito non raggiungibile",
            "body": [
                "Il sito del museo non è raggiungibile da questa mattina",
                "La pagina principale non si carica",
                "Il sito risulta offline"
            ],
            "priority": {"bassa": 0.0, "media": 0.3, "alta": 0.7}
        }
    ],

    "Commerciale": [
        {
            "title": "Prenotazione gruppo",
            "body": [
                "Vorrei prenotare una visita per un gruppo numeroso",
                "Desidero informazioni per una prenotazione di gruppo",
                "È possibile organizzare una visita per più persone?"
            ],
            "priority": {"bassa": 0.3, "media": 0.6, "alta": 0.1}
        },
        {
            "title": "Prezzi biglietti",
            "body": [
                "Qual è il costo del biglietto?",
                "Sono previste riduzioni sul prezzo del biglietto?",
                "Potrei avere informazioni sui prezzi?"
            ],
            "priority": {"bassa": 0.7, "media": 0.3, "alta": 0.0}
        },
        {
            "title": "Visite per scuole",
            "body": [
                "Organizzate visite didattiche per scuole?",
                "Ci sono percorsi dedicati alle scuole?",
                "Vorrei informazioni sulle visite scolastiche"
            ],
            "priority": {"bassa": 0.4, "media": 0.5, "alta": 0.1}
        }
    ]
}

def weighted_choice(weights):
    labels, probs = zip(*weights.items())
    return random.choices(labels, probs)[0]

rows = []

for i in range(NUM_TICKETS):
    category = random.choice(list(TICKETS.keys()))
    intent = random.choice(TICKETS[category])

    rows.append({
        "id": i + 1,
        "title": intent["title"],
        "body": random.choice(intent["body"]),
        "category": category,
        "priority": weighted_choice(intent["priority"])
    })

df = pd.DataFrame(rows)
df.to_csv("01_ticket_museo_le_nuove_ml.csv", index=False)

print("Dataset semanticamente coerente generato (500 ticket)")
