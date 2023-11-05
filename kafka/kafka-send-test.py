from kafka import KafkaProducer

def main():
    # Create a Kafka producer instance with the provided configuration
    producer = KafkaProducer(bootstrap_servers='localhost:9094')

    # Produce a message to the Kafka topic (replace 'your_topic_name' with your topic)
    producer.send('tim-topic', key=b'tim-key', value=b'Hello I am Tim Lam')

    # Ensure all messages are sent
    producer.flush()

if __name__ == '__main__':
    main()
