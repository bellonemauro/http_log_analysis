
# prima fase di conversione


# import json
# import ijson

# with open(r"path/to/my/file", "r") as infile, open("public_v2.ndjson", "w") as outfile:
#     for key, value in ijson.kvitems(infile, ''):
#         value["log_id"] = key
#         outfile.write(json.dumps(value) + "\n")


from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from detect import entropy
from detect import has_suspicious_request_len
import duckdb
import pandas as pd

con = duckdb.connect()

df = con.execute("""
    SELECT *
    FROM read_json_auto('public_v2.ndjson')
    LIMIT 100000
""").df()

df.info()


attacks = [
    {
        "referrer": "http://search.lib.auth.gr/Search/<script>alert('XSS')</script>",
        "request": "search.lib.auth.gr:80 185.22.12.44 - - [01/Mar/2018:00:00:15 +0200] GET /Search?q=<script>alert('XSS')</script> HTTP/1.1 400 1200 http://search.lib.auth.gr/Search/<script>alert('XSS')</script> Mozilla/5.0",
        "method": "GET",
        "resource": "/Search?q=<script>alert('XSS')</script>",
        "bytes": "1200",
        "response": "400",
        "ip": "185.22.12.44",
        "useragent": "Mozilla/5.0",
        "timestamp": "2018-02-28T22:00:15.000Z",
        "log_id": "attack_xss"
    },
    {
        "referrer": "http://evil.com",
        "request": "search.lib.auth.gr:80 45.77.88.12 - - [01/Mar/2018:00:00:20 +0200] GET /login?user=admin' OR 1=1-- HTTP/1.1 500 800 http://evil.com Mozilla/5.0",
        "method": "GET",
        "resource": "/login?user=admin' OR 1=1--",
        "bytes": "800",
        "response": "500",
        "ip": "45.77.88.12",
        "useragent": "Mozilla/5.0",
        "timestamp": "2018-02-28T22:00:20.000Z",
        "log_id": "attack_sql"
    },
    {
        "referrer": "-",
        "request": "search.lib.auth.gr:80 91.200.12.77 - - [01/Mar/2018:00:00:25 +0200] GET /../../../../etc/passwd HTTP/1.1 403 300 - curl/7.58.0",
        "method": "GET",
        "resource": "/../../../../etc/passwd",
        "bytes": "300",
        "response": "403",
        "ip": "91.200.12.77",
        "useragent": "curl/7.58.0",
        "timestamp": "2018-02-28T22:00:25.000Z",
        "log_id": "attack_traversal"
    },
    {
        "referrer": "-",
        "request": "search.lib.auth.gr:80 203.0.113.45 - - [01/Mar/2018:00:00:30 +0200] GET /wp-admin/install.php HTTP/1.1 404 250 - sqlmap/1.7",
        "method": "GET",
        "resource": "/wp-admin/install.php",
        "bytes": "250",
        "response": "404",
        "ip": "203.0.113.45",
        "useragent": "sqlmap/1.7",
        "timestamp": "2018-02-28T22:00:30.000Z",
        "log_id": "attack_scanner"
    }
]

df = pd.concat([df, pd.DataFrame(attacks)], ignore_index=True)


# timestamp in formato coerente

df["timestamp"] = pd.to_datetime(df["timestamp"])


# lughezza della resource richiesta

df["resource_len"] = df["resource"].str.len()


# da valutare se devo fare un primo controllo anche io sui quartili

df = has_suspicious_request_len(df)


# troppi special characters
df["special_chars"] = df["request"].str.count(r"[<>\'\"%;(){}]*+@")
df["special_chars"].value_counts()


# directory traversale
df["has_traversal"] = df["request"].str.contains(r"\.\./", regex=True, na=False).astype(int)
df["has_traversal"].value_counts()


# sql injection

df["has_sql_keywords"] = df["request"].str.contains(
    r"union|select|insert|drop|or 1=1|--|admin",
    case=False,
    na=False
).astype(int)

df["has_sql_keywords"].value_counts()


# script injection
df["has_script_tag"] = df["request"].str.contains(
    r"<script>",
    case=False,
    na=False
).astype(int)
df["has_script_tag"].value_counts()


# possibile scanner da parte di bot ostili
df["scanner_agent"] = df["useragent"].str.contains(
    r"sqlmap|nikto|curl|nmap|dirbuster",
    case=False,
    na=False
).astype(int)
df["useragent"].value_counts()


# status code di errore
df["is_error"] = (df["response"].astype(int) >= 400).astype(int)

df['is_error'].value_counts()


# numero di query eccessivo
df["num_params"] = df["resource"].str.count("=")
df["num_params"].value_counts()


# numero di richieste per IP
df["requests_per_ip"] = df.groupby("ip")["ip"].transform("count")


df["request_entropy"] = df["request"].apply(lambda x: entropy(str(x)))


df['bytes_num'] = pd.to_numeric(df['bytes'], errors='coerce').fillna(0)
resource_avg_bytes = df.groupby('resource')['bytes_num'].transform('mean')

df['bytes_ratio'] = df['bytes_num'] / (resource_avg_bytes + 1)
df['bytes_ratio'].value_counts()


# DA QUI IN POI ABBIAMO L'ALGORITMO DI ISOLATION FOREST
# 1. L'algoritmo non accetta stringhe o date. Dobbiamo isolare solo le colonne numeriche.
#


features_numeriche = [
    'resource_len', 'num_params', 'requests_per_ip', 'request_entropy', 'bytes_num',
    'bytes_ratio'
]

features_boolean = [
    'has_suspicious_request_len', 'special_chars',
    'has_traversal', 'has_sql_keywords', 'has_script_tag',
    'scanner_agent', 'is_error',
]


# Definiamo le colonne che l'algoritmo deve "studiare"
features = [
    'resource_len', 'has_suspicious_request_len', 'special_chars',
    'has_traversal', 'has_sql_keywords', 'has_script_tag',
    'scanner_agent', 'is_error', 'num_params',
    'requests_per_ip', 'request_entropy', 'bytes_num',
    'bytes_ratio'
]

X = df[features]


# 2. Configurazione e Addestramento
# L'Isolation Forest funziona creando alberi decisionali casuali: le anomalie vengono "isolate" più velocemente.
#


# contamination è la % stimata di attacchi nel dataset (es. 0.1% = 0.001)
model = IsolationForest(n_estimators=100, contamination=0.03, random_state=42)

# Addestramento
df['anomaly_score'] = model.fit_predict(X)


# Risultato: -1 significa Anomalia, 1 significa Traffico Normale


# 3. Interpretazione dei risultati
#
#


# Punteggio grezzo: valori negativi = più anomalo
df['scores'] = model.decision_function(X)

# per vedere se gli attacchi sono stati classificati come tali
df[df['log_id'].str.contains('attack')][['log_id', 'anomaly_score', 'scores']]


# 4. Verifica dei Falsi Positivi


# Mostra le 10 richieste più sospette del dataset
top_anomalies = df.sort_values(by='scores').head(10)
top_anomalies[['request', 'scores']]

# Gli attacchi che ho inserito io non sono in questa top... sono bot di google


# Dato che ho inserito degli attacchi manualmente (attack_xss, attack_sql, ecc.), posso calcolare una Matrice di Confusione semplificata per dimostrare quanto l'algoritmo è efficace nel trovare "l'ago nel pagliaio" che ho nascosto io.
#


df['is_attack_real'] = df['log_id'].str.contains('attack').astype(int)

df['is_attack_predicted'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)

cm = confusion_matrix(df['is_attack_real'], df['is_attack_predicted'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normale', 'Attacco'],
            yticklabels=['Normale', 'Attacco'])
plt.xlabel('Predizione Modello')
plt.ylabel('Realtà (Ground Truth)')
plt.title('Matrice di Confusione')
plt.show()

# TODO -> Isolare i bot di google che creano parecchio rumore di fondo


print(classification_report(df['is_attack_real'], df['is_attack_predicted']))


test_attacks = df[df['log_id'].str.contains('attack')]
test_attacks[['log_id', 'anomaly_score', 'scores']]
