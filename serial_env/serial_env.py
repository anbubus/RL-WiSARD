
import time
import serial
from typing import Optional, Callable
import threading
import re
import numpy as np
from gymnasium import spaces
import gymnasium as gym
# arduino = serialApp()
# arduino.serialPort.port = 'COM4'
BAUD_RATE = 115200
# arduino.serialPort.timeout=0.01
# arduino.closeSerial()
# arduino.updatePorts()


class SerialReader(threading.Thread):
    PATTERN = re.compile(r"R:([-\d.]+),P:([-\d.]+),Y:([-\d.]+)")

    def __init__(self, port: str, baud_rate:int, timeout:float = 1.0):
        super().__init__(daemon=True)

        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.connection: Optional[serial.Serial] = None
        self.connected = False

        self.roll    = 0.0
        self.pitch   = 0.0
        self.yaw     = 0.0
        
        self.error   = ""
        self._lock   = threading.Lock()
        self._stop_event = threading.Event()

    def connect(self)->bool:
        try:
            self.connection = serial.Serial(self.port, self.baud_rate, self.timeout)
            self.connected = True
            self.start()
            print(f"[Serial] Conectado em {self.port} @ {self.baud_rate} baud")

            return True
        
        except Exception as e:
            self.connected = False
            self.error = str(e)
            print(f"Serial ERROR - {self.error}")

            return False

    def disconnect(self):
        self._stop_event.set()
        self.join(timeout=3)
        if self.connected:
            self.connection.close()
            self.connected = False
        print("Connection closed.")

    def run(self):
        if self.connected:
            while not self._stop_event.is_set():
                try:
                    if self.connection.in_waiting:
                        line = self.connection.readline().decode("utf-8", errors="ignore").strip()
                        m = self.PATTERN.search(line)
                        if m:
                            with self._lock:
                                self.roll  = float(m.group(1))
                                self.pitch = float(m.group(2))
                                self.yaw   = float(m.group(3))
                except serial.SerialException as e:
                    print(e)
                    break
                time.sleep(0.01)

    def write(self, data:str):
        if not self.connected:
            return False
        data_ = (data + "\n")
        with self._lock:
            try:
                self.connection.write(data_.encode("utf-8"))
                return True
            except serial.SerialException as e:
                print(e)
                return False 

    def get_angles(self):
        with self._lock:
            return self.roll, self.pitch, self.yaw

class CustomEnv(gym.Env):
    def __init__(self, max_steps:int):
        self.PORT = "COM4"
        self.BAUDRATE = 115200
        self.ser = SerialReader(self.PORT, self.BAUDRATE)

        self.action_space = spaces.Discrete(2)

        low = np.array([-20] * 3 + [-180] * 3, dtype=np.float32)
        high = np.array([20] * 3 + [180] * 3, dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.step_count = 0
        self.max_steps = max_steps ##500000

    def serial_start(self):
        try:
            self.ser.start()
        except:
            print("ERROR: ", self.ser.error)
        

    def step(self, action):
        if self.ESP.connected:
            self.step_count += 1

            self.send_action(action)
            obs = self.get_observation()
            reward = self.get_reward(obs)

            terminated = self.is_terminated(obs)
            truncated = self.step_count > self.max_steps


            return obs, reward, terminated, truncated, {}


    def send_action(self, action)->bool:
        dir = "EEE" if action==0 else "DDD"
        return self.ser.write(dir)
            
    def get_observation(self):
        obs = self.ser.read()
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs.astype(np.float32)

    def get_reward(self, obs):
        roll, pitch = obs[3], obs[4]
        angle_penalty = (roll ** 2 + pitch ** 2) / (30.0 ** 2)
        reward = 1.0 - angle_penalty

        return float(np.clip(reward, -1.0, 1.0))
    
    def is_termitated(self, obs):
        roll, pitch = obs[3], obs[4]
        return bool(abs(roll) > 30.0 or abs(pitch) > 30.0)

    def restart(self):
        super().reset(seed=seed)
        self._step_count = 0
        obs = self._get_observation()
        return obs, {}
        

    def end(self):
        pass

   















# serial = serialApp()
# serial.serialPort.port = 'COM4'
# serial.serialPort.baudrate = 115200
# serial.serialPort.timeout=.1
# serial.closeSerial()
# serial.updatePorts()

# serial.connectSerial()
# time.sleep(0.5) 
# for i in range(10):
#     #num = input("Press a key: ")
#     serial.writeSerial(i)
#     serial.readSerial()
# serial.closeSerial()

# Importing Libraries 
# import serial 
# import time 
# arduino = serial.Serial(port='COM4', baudrate=115200, timeout=.1) 
# def write_read(x): 
#     arduino.write(bytes(x, 'utf-8')) 
#     time.sleep(0.05) 
#     data = arduino.readline() 
#     return data 
# while True: 
#     num = input("Enter a number: ") # Taking input from user 
#     value = write_read(num) 
#     print(value.decode()) # printing the value 
