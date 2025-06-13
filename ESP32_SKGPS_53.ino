#include <Arduino.h>

// --- Pins ---
#define LED_PIN      4
#define BUZZER_PIN   15
#define BUTTON_PIN   18

// --- Serial port for GPS (uses UART2) ---
#define GPS_RX       16
#define GPS_TX       17

void setup() {
  // Serial for debug output via USB cable
  Serial.begin(115200);
  delay(1000);
  Serial.println("System started.");

  // Serial2 for GPS communication
  Serial2.begin(9600, SERIAL_8N1, GPS_RX, GPS_TX);

  // Setup LED and button
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP); // Active LOW button

  // Setup buzzer using PWM on channel 0
  ledcSetup(0, 2000, 8);             // Channel 0, 2kHz, 8-bit
  ledcAttachPin(BUZZER_PIN, 0);      // Attach pin to channel
}

void loop() {
  // Turn on LED and buzzer
  digitalWrite(LED_PIN, HIGH);
  ledcWrite(0, 127); // Medium volume

  // Check button press (LOW when pressed)
if (digitalRead(BUTTON_PIN) == LOW) {
  Serial.println("Button pressed! Reading GPS...");
  
  // Read and forward full NMEA lines from GPS
  while (Serial2.available()) {
    char c = Serial2.read();
    Serial.write(c);  // Forward raw GPS data to PC
  }

  delay(1000);  // Debounce
}

  delay(100); // Loop delay
}
