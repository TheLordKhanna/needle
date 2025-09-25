/*
this is motor control for six motors. this is not my original work, it is simply
modified from the dynamixel2arduino library for six XL430 motor.
*/

#include <Dynamixel2Arduino.h>

#if defined(ARDUINO_AVR_UNO) || defined(ARDUINO_AVR_MEGA2560)
  #include <SoftwareSerial.h>
  SoftwareSerial soft_serial(7, 8);
  #define DXL_SERIAL   Serial
  #define DEBUG_SERIAL soft_serial
  const int DXL_DIR_PIN = 2;
#elif defined(ARDUINO_SAM_DUE)
  #define DXL_SERIAL   Serial
  #define DEBUG_SERIAL SerialUSB
  const int DXL_DIR_PIN = 2;
#elif defined(ARDUINO_SAM_ZERO)
  #define DXL_SERIAL   Serial1
  #define DEBUG_SERIAL SerialUSB
  const int DXL_DIR_PIN = 2;
#elif defined(ARDUINO_OpenCM904)
  #define DXL_SERIAL   Serial3
  #define DEBUG_SERIAL Serial
  const int DXL_DIR_PIN = 22;
#elif defined(ARDUINO_OpenCR)
  #define DXL_SERIAL   Serial3
  #define DEBUG_SERIAL Serial
  const int DXL_DIR_PIN = 84;
#elif defined(ARDUINO_OpenRB)
  #define DXL_SERIAL   Serial1
  #define DEBUG_SERIAL Serial
  const int DXL_DIR_PIN = -1;
#else
  #define DXL_SERIAL   Serial1
  #define DEBUG_SERIAL Serial
  const int DXL_DIR_PIN = 2;
#endif

const uint8_t ids[] = {1, 2, 3, 4, 5, 6};
const size_t motorCount = sizeof(ids) / sizeof(ids[0]);
const float DXL_PROTOCOL_VERSION = 2.0;
Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
using namespace ControlTableItem;

void setup() {
  DEBUG_SERIAL.begin(115200);
  while (!DEBUG_SERIAL);

  dxl.begin(57600);
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);

  for (size_t i = 0; i < motorCount; ++i) {
    uint8_t id = ids[i];
    dxl.ping(id);
    dxl.torqueOff(id);
    dxl.setOperatingMode(id, OP_EXTENDED_POSITION);
    dxl.torqueOn(id);
    dxl.writeControlTableItem(PROFILE_VELOCITY, id, 50);
  }
}

void loop() {
  int32_t rawTargets[motorCount] = {
    -4365,
    -512+79,
    5534,
    512-295,
    374,
    1169
  };

  for (size_t i = 0; i < motorCount; ++i) {
    dxl.setGoalPosition(ids[i], rawTargets[i]);
  }

  bool busy;
  do {
    busy = false;
    for (size_t i = 0; i < motorCount; ++i) {
      int32_t pos = dxl.getPresentPosition(ids[i]);
      if (abs(rawTargets[i] - pos) > 20) busy = true;
    }
  } while (busy);

  delay(1000);

  float degTargets[motorCount] = {
    (374/4096)*360,
    ((-512+79)/4096)*360,
    (5534/4096)*360,
    ((512-295)/4096)*360,
    (374/4096)*360,
    (1169/4096)*360
  };

  for (size_t i = 0; i < motorCount; ++i) {
    dxl.setGoalPosition(ids[i], degTargets[i], UNIT_DEGREE);
  }

  do {
    busy = false;
    for (size_t i = 0; i < motorCount; ++i) {
      float d = dxl.getPresentPosition(ids[i], UNIT_DEGREE);
      if (abs(degTargets[i] - d) > 2.0) busy = true;
    }
  } while (busy);

  delay(2000);
}
