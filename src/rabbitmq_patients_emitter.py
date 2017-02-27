import pika
import os


INPUT_FOLDER = '/Volumes/Files/git/kaggle-dataScienceBowl2017/data/stage1/'

credentials = pika.PlainCredentials(username='user', password='Qwer1234!')
connection = pika.BlockingConnection(pika.ConnectionParameters(host='192.168.0.7', credentials=credentials))
channel = connection.channel()
channel.queue_declare(queue='ds_images')


patients = os.listdir(INPUT_FOLDER)
patients.sort()
patients = [p for p in patients if not p.startswith(".")]

for patient in patients:
    channel.basic_publish(exchange='', routing_key='ds_images', body=patient)
    print("[x] Sent", patient)

connection.close()
