import pika
from preprocessing_worker import process_file
import json


credentials = pika.PlainCredentials(username='user', password='Qwer1234!')
parameters = pika.ConnectionParameters(host='192.168.0.7', credentials=credentials)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()
channel.queue_declare(queue='ds_images')
channel.queue_declare(queue='ds_preprocessing_results')
channel.queue_declare(queue='ds_errors')
print('[*] Waiting for messages. To exit press CTRL+C')


def send_results(patient, hu_in_mean, hu_resample_image_mean, hu_out_image_mean):
    data = {
        'patient': patient,
        'hu_in_mean': hu_in_mean,
        'hu_resample_image_mean': hu_resample_image_mean,
        'hu_out_image_mean': hu_out_image_mean
    }
    channel.basic_publish(exchange='', routing_key='ds_preprocessing_results', body=json.dumps(data))


def send_error(patient):
    channel.basic_publish(exchange='', routing_key='ds_errors', body=patient)


def callback(ch, method, properties, body):
    print("[x] Received %r" % body.decode("utf-8"))

    try:
        hu_in_mean, hu_resample_image_mean, hu_out_image_mean = process_file(body.decode("utf-8"))
        send_results(body.decode("utf-8"), hu_in_mean, hu_resample_image_mean, hu_out_image_mean)
    except Exception as e:
        send_error(body.decode("utf-8"))
        print(str(e))
    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(callback, queue='ds_images')
channel.start_consuming()
