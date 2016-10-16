// Smartphone Opthalmoscope Firmware //
 
// Constants
const int IR_LED_1 = 2;
const int WHITE_LED = 3;
const int IR_LED_2 = 4;
const int RED_STATUS_LED = 8; // On when IR LED is on
const int GREED_STATUS_LED = 9; // Always on
const int BUTTON_PIN = 13;
 
// Variables
int BUTTON_PUSH_COUNTER = 0; //counter for the number of button presses
int BUTTON_STATE = 0;// current state of the button
int LAST_BUTTON_STATE = 0;// previous state of the button
 
void setup() {
 
  // initialise the button pin as a input:
  pinMode(BUTTON_PIN, INPUT);
 
  // initialise pins as an output:
  pinMode(IR_LED_1, OUTPUT);
  pinMode(WHITE_LED, OUTPUT);
  pinMode(IR_LED_2, OUTPUT);
  pinMode(RED_STATUS_LED, OUTPUT);
  pinMode(GREED_STATUS_LED, OUTPUT);
 
  // initialize serial communication:
  Serial.begin(9600);
 
}
 
void loop() {
  // read the pushbutton input pin:
  BUTTON_STATE = digitalRead(BUTTON_PIN);
 
  // compare the buttonState to its previous state
 
  // compare the buttonState to its previous state
  if (BUTTON_STATE != LAST_BUTTON_STATE) {
    // if the state has changed, increment the counter
    if (BUTTON_STATE == HIGH) {
      // if the current state is HIGH then the button
      // wend from off to on:
      BUTTON_PUSH_COUNTER++;
      Serial.println("on");
      Serial.print("number of button pushes:  ");
      Serial.println(BUTTON_PUSH_COUNTER);
      /// Reset counter after the 4th state
      if (BUTTON_PUSH_COUNTER > 4){
        BUTTON_PUSH_COUNTER = 0;
      }
 
    } else {
      // if the current state is LOW then the button
      // wend from on to off:
      Serial.println("off");
    }
    // Delay a little bit to avoid bouncing
    delay(50);
  }
  // save the current state as the last state,
  //for next time through the loop
  LAST_BUTTON_STATE = BUTTON_STATE;
 
  // White LED State
  if (BUTTON_PUSH_COUNTER == 1)
  {
      digitalWrite(WHITE_LED, LOW);
   
  }
  // IR LED State
  else if (BUTTON_PUSH_COUNTER == 2)
  {
      digitalWrite(RED_STATUS_LED, LOW);
      digitalWrite(IR_LED_1, LOW);
      digitalWrite(IR_LED_2, LOW);
 
  }
  // All on State
  else if (BUTTON_PUSH_COUNTER == 3)
  {
      digitalWrite(IR_LED_1, HIGH);
      digitalWrite(WHITE_LED, HIGH);
      digitalWrite(IR_LED_2, HIGH);
      digitalWrite(RED_STATUS_LED, HIGH);
      digitalWrite(GREED_STATUS_LED, HIGH);
  }
  // All off State
  else if (BUTTON_PUSH_COUNTER = 4)
  {
      digitalWrite(IR_LED_1, LOW);
      digitalWrite(WHITE_LED, LOW);
      digitalWrite(IR_LED_2, LOW);
      digitalWrite(RED_STATUS_LED, LOW);
      digitalWrite(GREED_STATUS_LED, LOW);
  }
 
}
