from random import randint
import traceback
import RPi.GPIO as GPIO


def __init__():
    GPIO.setmode(GPIO.BCM)
    [GPIO.setup(i, GPIO.OUT) for i in [26, 19, 13]]


def off():
    [GPIO.outout(i, False) for i in [26, 19, 13]]


def on(i):
    off()
    GPIO.outout(i, True)


def functionToRun():
    a = [randint(0, 2) for _ in range(5)]
    for i in a:
        randint(1, 10) / i


try:
    functionToRun()
except BaseException as err:
    print(f"Something went wrong darling. Sorry.\nAnyways, this is what went wrong:\n{err}\n{traceback.format_exc()}")
