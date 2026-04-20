import time
import board
import neopixel
import RPi.GPIO as GPIO

# ====== NeoPixel settings ======
PIXEL_PIN = board.D18     # GPIO18 (PWM0)
NUM_PIXELS = 12           # 必要なら変更
BRIGHTNESS = 0.15         # 15% 安全寄り
PIXEL_ORDER = neopixel.GRB

pixels = neopixel.NeoPixel(
    PIXEL_PIN,
    NUM_PIXELS,
    brightness=BRIGHTNESS,
    auto_write=False,
    pixel_order=PIXEL_ORDER
)

# ====== IR LED (MOSFET Gate) settings ======
IR_PIN = 23               # GPIO23
PWM_FREQ = 1000           # 1kHz
IR_DUTY_ON = 20           # 20% (最初は低め)
IR_DUTY_OFF = 0

GPIO.setmode(GPIO.BCM)
GPIO.setup(IR_PIN, GPIO.OUT)

ir_pwm = GPIO.PWM(IR_PIN, PWM_FREQ)
ir_pwm.start(IR_DUTY_OFF)


def set_all(color):
    pixels.fill(color)
    pixels.show()


def chase(color, wait=0.05):
    for i in range(NUM_PIXELS):
        pixels.fill((0, 0, 0))
        pixels[i] = color
        pixels.show()
        time.sleep(wait)


try:
    print("=== Full system test start ===")

    # 1) IR を低デューティでON（常時）
    print(f"IR PWM ON: {IR_DUTY_ON}% @ {PWM_FREQ}Hz")
    ir_pwm.ChangeDutyCycle(IR_DUTY_ON)
    time.sleep(0.3)

    # 2) NeoPixel: 基本色を順に
    print("NeoPixel: RED")
    set_all((255, 0, 0))
    time.sleep(1.0)

    print("NeoPixel: GREEN")
    set_all((0, 255, 0))
    time.sleep(1.0)

    print("NeoPixel: BLUE")
    set_all((0, 0, 255))
    time.sleep(1.0)

    print("NeoPixel: WHITE (low brightness)")
    set_all((255, 255, 255))
    time.sleep(1.0)

    # 3) 走査点灯（配線・データの安定性確認）
    print("NeoPixel: chase test")
    chase((255, 0, 0), 0.05)
    chase((0, 255, 0), 0.05)
    chase((0, 0, 255), 0.05)

    # 4) IR PWM をON/OFFして干渉を見る
    print("IR PWM toggle test")
    for _ in range(5):
        ir_pwm.ChangeDutyCycle(IR_DUTY_ON)
        time.sleep(0.4)
        ir_pwm.ChangeDutyCycle(IR_DUTY_OFF)
        time.sleep(0.4)

    # 5) 最後に消灯
    print("All OFF")
    set_all((0, 0, 0))
    ir_pwm.ChangeDutyCycle(IR_DUTY_OFF)
    time.sleep(0.2)

    print("=== Test done ===")

finally:
    # 安全に後始末
    try:
        set_all((0, 0, 0))
    except Exception:
        pass

    try:
        ir_pwm.ChangeDutyCycle(0)
        ir_pwm.stop()
    except Exception:
        pass

    GPIO.cleanup()