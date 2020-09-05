import socket
import requests


def safe_get(func, *args, **kwargs):
    result = None
    while result is None:
        try:
            result = func(*args, **kwargs)
        except requests.exceptions.RequestException:
            print('ConnectionError!')
        except socket.error:
            print('Time Out!')
        except KeyboardInterrupt:
            exit()
    return result