from pymongo import MongoClient
uri = "mongodb+srv://abhishekguntuk01_db_user:Abhi22@cluster0.lged9yj.mongodb.net/?appName=Cluster0"
client = MongoClient(uri)
try:
    client.admin.command("ping")
    print("Connected successfully")
    client.close()

except Exception as e:
    raise Exception(
        "The following error occurred: ", e)