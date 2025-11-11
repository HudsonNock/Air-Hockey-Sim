
//SYSTEM ID CODE
#include <Arduino.h>
#include <SPI.h>
#include <cmath>

#define ENC_CHIP_SELECT_LEFT PB12
#define ENC_CHIP_SELECT_RIGHT PA4

#define LEFT_MOTOR_PWM_PIN PB_6
#define RIGHT_MOTOR_PWM_PIN PB_7
#define LEFT_MOTOR_DIR_PIN PB5
#define RIGHT_MOTOR_DIR_PIN PB4

#define DUTY_CYCLE_CONVERSION 1024 // Accepted duty cycle values are 0-1024
#define PWM_FREQ_HZ 10000
#define ROLLOVER_ANGLE_DEGS 180
#define MALLET_RADIUS 0.1011/2

using namespace std;

int mode = 5; //1: homing only, 2: feedback only FP, 3: feedback only path, 4: feedforward only, 5: feedback + feedforward

float PULLEY_RADIUS = 0.035755;
array<float,2> revolutions = {0,0};
array<float,2> offset = {0,0};

array<float,2> angle = {0,0};

array<float,2> p_encoder = {0,0};
array<float,2> encoder = {0,0};

array<float,2> xy_pos = {0,0};
array<float,2> pwm_vals = {0,0};

float height = 1.993;
float width = 0.992;

array<float,2> wall = {0.07974, 0.08022}; // thickness in x and y, mallet R 0.101553/2

float V_max = 24;

array<float,2> dts = {0,0};

double vt_1[2], vt_2[2], Vf[2], t_init;
struct __attribute__((packed)) Packet16 {
  int16_t pos_q[2];
  int16_t pwm_q[2];
  int16_t dt_q;
  uint8_t checksum;
};

Packet16 pkt;

array<float,2> Vxy = {0,0};
array<float,2> V = {0,0};

u_int16_t serial_response; // incoming byte from the SPI
int chips[2] = {ENC_CHIP_SELECT_LEFT, ENC_CHIP_SELECT_RIGHT};

double f(double x, int i);
double g(double tms, int i);
void expected_position(double t);
void apply_voltage(double t);
void setup_coefficients();
void read_motor_angles();
void theta_to_xy();
void xy_to_theta(array<float,2> xy);
void set_motor_pwms(float left, float right);
void home_table();
void send_motor_pos();
bool update_goal();
void empty_buffer();

int counter = 0;

void setup() {
  Serial.begin(460800);
  Serial.setTimeout(0);
  SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE1));

  pinMode(ENC_CHIP_SELECT_LEFT, OUTPUT);
  pinMode(ENC_CHIP_SELECT_RIGHT, OUTPUT); 

  digitalWrite(ENC_CHIP_SELECT_LEFT, HIGH);
  digitalWrite(ENC_CHIP_SELECT_RIGHT, HIGH);

  pinMode(LEFT_MOTOR_PWM_PIN, OUTPUT);
  pinMode(RIGHT_MOTOR_PWM_PIN, OUTPUT);
  pinMode(LEFT_MOTOR_DIR_PIN, OUTPUT);
  pinMode(RIGHT_MOTOR_DIR_PIN, OUTPUT);
  
  pwm_start(LEFT_MOTOR_PWM_PIN, PWM_FREQ_HZ, 0, RESOLUTION_10B_COMPARE_FORMAT);
  pwm_start(RIGHT_MOTOR_PWM_PIN, PWM_FREQ_HZ, 0, RESOLUTION_10B_COMPARE_FORMAT);
  digitalWrite(LEFT_MOTOR_DIR_PIN, LOW);
  digitalWrite(RIGHT_MOTOR_DIR_PIN, LOW);

  delay(1000);
  empty_buffer();
  while (Serial.available() == 0) {
    continue;
  }
  delay(100);
  Serial.println("r");
  empty_buffer();
  while (Serial.available() == 0) {
    continue;
  }
  delay(100);
  empty_buffer();
  home_table(); // remove calibration device

  delay(100); //turn on Power
  empty_buffer();
  while (Serial.available() == 0) {
    read_motor_angles();
  }
  Serial.println("r");

  empty_buffer();
  delay(100);
  read_motor_angles();
  send_motor_pos();
  do {
    while (Serial.available() < 54) {
      read_motor_angles();
    }
  } while (!update_goal());

  t_init = micros() / 1e6;

  dts[0] = micros() / 1e3;
  dts[1] = dts[0];
}

void empty_buffer() {
  while (Serial.available()) {
    Serial.read();
  }
}

bool update_goal() {
  if (Serial.available() >= 54) {
    char start_char = Serial.read();
    if (start_char != '\n') {
      while (Serial.available() && Serial.read() != '\n');
      return false;
    }
    

    uint8_t buffer[52];
    Serial.readBytes(buffer,52);
    if (!Serial.available() || Serial.read() != '\n') {
      return false;
    }

    int32_t vt_1_0 = *(int32_t*)(buffer + 0);
    int32_t vt_1_1 = *(int32_t*)(buffer + 4);
    int32_t vt_2_0 = *(int32_t*)(buffer + 8);
    int32_t vt_2_1 = *(int32_t*)(buffer + 12);
    int32_t Vf_0 = *(int32_t*)(buffer + 16);
    int32_t Vf_1 = *(int32_t*)(buffer + 20);
    int32_t C2_0 = *(int32_t*)(buffer + 24);
    int32_t C2_1 = *(int32_t*)(buffer + 28);
    int32_t C3_0 = *(int32_t*)(buffer + 32);
    int32_t C3_1 = *(int32_t*)(buffer + 36);
    int32_t C4_0 = *(int32_t*)(buffer + 40);
    int32_t C4_1 = *(int32_t*)(buffer + 44);
    int32_t sum = *(int32_t*)(buffer + 48);
    
    int32_t calculated_sum = vt_1_0 ^ vt_1_1 ^ vt_2_0 ^ vt_2_1 ^ Vf_0 ^ Vf_1 ^ C2_0 ^ C2_1 ^ C3_0 ^ C3_1 ^ C4_0 ^ C4_1;
    if (sum != calculated_sum) {
      return false;
    }

    t_init = micros() / 1e6;
    vt_1[0] = vt_1_0 / 10000.0;
    vt_1[1] = vt_1_1 / 10000.0;
    vt_2[0] = vt_2_0 / 10000.0;
    vt_2[1] = vt_2_1 / 10000.0;
    Vf[0] = Vf_0 / 10000.0;
    Vf[1] = Vf_1 / 10000.0;

    return true;
  }
  else if (mode == 1) {
    t_init = micros() / 1e6;
    return true;
  }
  return false;
}

uint8_t compute_checksum(const uint8_t *data, size_t len){
  uint8_t c = 0;
  for(size_t i=0; i<len; i++) c ^= data[i];
  return c;
}

void send_motor_pos() {
  auto quant = [&](float x, float xmin, float xmax){
    float y = 2*(x - xmin)/(xmax - xmin) - 1;
    int16_t s = int16_t(round(y * 32767));
    if ((s & 0x000F) == 0x0F) {
        s -= 1;  // Decrease by 1 to make last nibble 0xE
    }
    return s;
  };

  dts[1] = dts[0];
  dts[0] = micros() / 1e3;
  float dt = dts[0] - dts[1];

  pkt.pos_q[0] = quant(xy_pos[0], -0.5, 2);  
  pkt.pos_q[1] = quant(xy_pos[1], -0.5, 2);
  pkt.pwm_q[0] = quant(pwm_vals[0], -1.1, 1.1);
  pkt.pwm_q[1] = quant(pwm_vals[1], -1.1, 1.1);
  pkt.dt_q = quant(dt, 0, 2);

  // checksum over the six int16s (12 bytes)
  pkt.checksum = compute_checksum((uint8_t*)&pkt, sizeof(pkt) - 1);

  Serial.write(0xFF);
  Serial.write(0xFF);
  Serial.write(0xFF);
  Serial.write((uint8_t*)&pkt, sizeof(pkt));
  Serial.write(0x55);
}

void loop() {
  update_goal();

  read_motor_angles();
  float dt = micros() / 1e6 - t_init;
  apply_voltage(dt);

  send_motor_pos();
}

void apply_voltage(double t) {

  Vxy[0] = 0;
  Vxy[1] = 0;
  V[0] = 0;
  V[1] = 0;

  if (mode == 5 || mode == 4) {
    if (t<vt_1[0]) {
        Vxy[0] = Vf[0];
    } else if (t < vt_2[0]){
        Vxy[0] = -Vf[0];
    }

    if (t<vt_1[1]) {
        Vxy[1] = Vf[1];
    } else if (t < vt_2[1]) {
        Vxy[1] = -Vf[1]; 
    }

    V[0] = Vxy[0]/2 - Vxy[1]/2;
    V[1] = -Vxy[0]/2 - Vxy[1]/2;
  }

  if (V[0] > V_max) {
    V[0] = V_max;
  }
  else if (V[0] < -V_max) {
    V[0] = -V_max;
  }

  if (V[1] > V_max) {
    V[1] = V_max;
  }
  else if (V[1] < -V_max) {
    V[1] = -V_max;
  }

  if (abs(V[0]) < 1.5) {
    V[0] = 0;
  }

  if (abs(V[1]) < 1.5) {
    V[1] = 0;
  }
  
  set_motor_pwms(V[0]/24.0, V[1]/24.0);
}

void read_motor_angles() {
  p_encoder = encoder;

  digitalWrite(chips[0], LOW);
  serial_response = SPI.transfer16(0x3FFF);
  digitalWrite(chips[0], HIGH);
  encoder[0] = (serial_response & 0b0011111111111111) * 360.0 / 16384;

  if(encoder[0] - p_encoder[0] > ROLLOVER_ANGLE_DEGS) {
      revolutions[0] += 1;
  } else if(p_encoder[0] - encoder[0] > ROLLOVER_ANGLE_DEGS) {
      revolutions[0] -= 1;
  }

  digitalWrite(chips[1], LOW);
  serial_response = SPI.transfer16(0x3FFF);
  digitalWrite(chips[1], HIGH);
  encoder[1] = (serial_response & 0b0011111111111111) * 360.0 / 16384;

  if(encoder[1] - p_encoder[1] > ROLLOVER_ANGLE_DEGS) {
      revolutions[1] += 1;
  } else if (p_encoder[1] - encoder[1] > ROLLOVER_ANGLE_DEGS) {
      revolutions[1] -= 1;
  }

  angle[0] = -encoder[0] + 360.0 * revolutions[0] + offset[0];
  angle[1] = -encoder[1] + 360.0 * revolutions[1] + offset[1];
  theta_to_xy();
}

void theta_to_xy() {
  xy_pos[0] = (angle[0] - angle[1]) * PULLEY_RADIUS * PI / 360;
  xy_pos[1] = (-angle[0] - angle[1]) * PULLEY_RADIUS * PI / 360;
}

void set_motor_pwms(float left, float right) {
  pwm_vals[0] = left;
  if (left >= 0) {
      digitalWrite(LEFT_MOTOR_DIR_PIN, LOW);
  } else {
      digitalWrite(LEFT_MOTOR_DIR_PIN, HIGH);
  }
  pwm_start(LEFT_MOTOR_PWM_PIN, PWM_FREQ_HZ, floor(abs(left) * DUTY_CYCLE_CONVERSION), RESOLUTION_10B_COMPARE_FORMAT);

  pwm_vals[1] = right;
  if (right >= 0) {
      digitalWrite(RIGHT_MOTOR_DIR_PIN, LOW);
  } else {
      digitalWrite(RIGHT_MOTOR_DIR_PIN, HIGH);
  }
  pwm_start(RIGHT_MOTOR_PWM_PIN, PWM_FREQ_HZ, floor(abs(right) * DUTY_CYCLE_CONVERSION), RESOLUTION_10B_COMPARE_FORMAT);
}

void home_table() {
  //Home Y
  array<float,2> right_point;
  array<float,2> left_point;

  empty_buffer();

  while (Serial.available() == 0) {
    read_motor_angles();
  }
  read_motor_angles();
  right_point = angle;

  Serial.println("confirmation of right");

  empty_buffer();

  while (Serial.available() == 0) {
    read_motor_angles();
  }
  read_motor_angles();
  left_point = angle;

  PULLEY_RADIUS = (360/PI) * (width - 2*(MALLET_RADIUS + wall[1])) / (right_point[0] + right_point[1] - left_point[0] - left_point[1]);
  offset[1] = (360 * (2*MALLET_RADIUS+wall[0]+wall[1]) / (PULLEY_RADIUS * PI) + 2*right_point[1])/(-2);
  offset[0] = (360*(wall[0] - wall[1])/(PULLEY_RADIUS*PI) - 2*right_point[0])/2;

  Serial.println(PULLEY_RADIUS, 8);

  empty_buffer();

  //wait for calibration device to be removed
  while (Serial.available() == 0) {
    read_motor_angles();
  }

  Serial.println("starting");
  empty_buffer();
}
