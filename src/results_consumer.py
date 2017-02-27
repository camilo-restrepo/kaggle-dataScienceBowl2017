import pika
import json
import csv


OUTPUT_FOLDER = '/Volumes/Files/git/kaggle-dataScienceBowl2017/data/'
OUTPUT_FILE = OUTPUT_FOLDER + 'results.csv'
keys = ['patient', 'hu_in_mean', 'hu_resample_image_mean', 'hu_out_image_mean']

credentials = pika.PlainCredentials(username='user', password='Qwer1234!')
connection = pika.BlockingConnection(pika.ConnectionParameters(host='192.168.0.7', credentials=credentials))
channel = connection.channel()
channel.queue_declare(queue='ds_preprocessing_results')
print('[*] Waiting for messages. To exit press CTRL+C')


def callback(ch, method, properties, body):
    data = json.loads(body.decode('utf-8'))
    print("[x] Received", data['patient'])

    with open(OUTPUT_FILE, 'a') as csv_file:
        w = csv.DictWriter(csv_file, keys)
        w.writerow(data)

    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(callback, queue='ds_preprocessing_results')
channel.start_consuming()


