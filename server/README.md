# Federated Learning Server

This project implements a federated learning server using the TinyFL framework. The server is designed to connect with multiple clients, aggregate their model updates, and perform federated averaging to improve a global model.

## Project Structure

```
federated_learning_server
├── notebooks
│   └── interactive_server.ipynb  # Jupyter notebook for running the federated learning server
├── src
│   ├── server
│   │   ├── __init__.py            # Initialization file for the server module
│   │   └── fedavg.py              # Implements the federated averaging algorithm
│   └── utils
│       ├── __init__.py            # Initialization file for the utils module
│       └── mqtt_client.py          # Handles MQTT client functionality
├── config
│   └── server_config.json          # Configuration settings for the server
├── requirements.txt                 # Python package dependencies
└── README.md                        # Documentation for the project
```

## Setup Instructions

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required packages using pip:

   ```
   pip install -r requirements.txt
   ```

4. Configure the server settings in `config/server_config.json` as needed.

## Usage Guidelines

To run the federated learning server:

1. Open the `notebooks/interactive_server.ipynb` in Jupyter Notebook.
2. Follow the instructions in the notebook to initialize the server and wait for client connections.
3. When prompted, input 'y' to start training for 100 rounds.

## Additional Information

- Ensure that the MQTT broker is running and accessible as per the configuration in `server_config.json`.
- The server will log its activities, which can be monitored for debugging and performance evaluation.

For any issues or contributions, please refer to the project's issue tracker.