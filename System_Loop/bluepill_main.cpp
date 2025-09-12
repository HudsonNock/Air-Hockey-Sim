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
#define PULLEY_RADIUS 0.035686666463

using namespace std;

int mode = 5; //1: homing only, 2: feedback only FP, 3: feedback only path, 4: feedforward only, 5: feedback + feedforward

array<float,2> revolutions = {0,0};
array<float,2> offset = {0,0};

array<float,2> angle = {0,0};
array<float,2> p_angle = {0,0};
array<float,2> pp_angle = {0,0};

array<float,2> p_encoder = {0,0};
array<float,2> encoder = {0,0};

array<float,2> xy_pos = {0,0};
array<float,2> p_xy_pos = {0,0};
array<float,2> pp_xy_pos = {0,0};

array<float,2> xy_acc = {0,0};
array<float,2> xy_vel = {0,0};

array<float,2> xf = {0.5,0.5};
array<float,2> expected_xy_pos = {0.4, 0.52};

array<float,2> dt_angle = {0.1,0.1};
float past_time_angle = 0;

array<float,2> past_err = {0,0};
float past_time_err = 0;

array<float,2> theta_hat = {0,0};
array<float,2> err = {0, 0};

float current_time_err = 0;
float dt_err = 0;

array<float,2> accumulated_err = {0,0};

float height = 1.993;
float width = 0.992;

float kp = 0.1;
float ki = 0.001;
float kd = 0.005;

array<float,2> wall = {0.07974, 0.08022}; // thickness in x and y, mallet R 0.101553/2

float V_max = 20;

const double a1 = 3.579e-6;
const double a2 = 0.00571;
const double a3 = (0.0596 + 0.0467) / 2.0;
const double b1 = -1.7165e-6;
const double b2 = -0.002739;
const double b3 = 0.0;

double C5[2] = {a1 - b1, a1 + b1};
double C6[2] = {a2 - b2, a2 + b2};
double C7[2] = {a3 - b3, a3 + b3};

double CE[2][6] = {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}};
double ab[2][2] = {{0, 0}, {0, 0}};
double A2CE[2][3] = {{0, 0, 0}, {0, 0, 0}};
double A3CE[2][2] = {{0, 0}, {0, 0}};
double A4CE[2][2] = {{0, 0}, {0, 0}};

double vt_1[2], vt_2[2], Vf[2], C2[2], C3[2], C4[2];

double t_init = 0;
double dt = 0;

float dt_sum = 0;
array<float,2> new_xy_vel = {0,0};
array<float,2> new_xy_acc = {0,0};

float alpha_acc = 0.05;  // Adjust between 0.0 (no change) and 1.0 (instant change)
float alpha_vel = 0.5;

struct __attribute__((packed)) Packet16 {
  int16_t pos_q[2];
  int16_t vel_q[2];
  int16_t acc_q[2];
  uint8_t checksum;
};

Packet16 pkt;

array<float,2> Vxy = {0,0};
array<float,2> V = {0,0};

double eat, ebt, A_2, A_3, A_4, f_t, g_a1, g_a2;

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

  setup_coefficients();

  t_init = micros() / 1e6;
  vt_1[0] = 0.004;
  vt_1[1] = 0.005;
  vt_2[0] = 0.009;
  vt_2[1] = 0.006;
  Vf[0] = 13.8;
  Vf[1] = 1.8;
  C2[0] = 0.0008;
  C2[1] = 0.0003;
  C3[0] = 0.000009;
  C3[1] = 0.00007987;
  C4[0] = 0.0007654;
  C4[1] = 0.00002459;

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
  home_table();

  t_init = micros() / 1e6;

  past_time_angle = micros() / 1e6;
  past_time_err = micros() / 1e6;
}

void empty_buffer() {
  while (Serial.available()) {
    Serial.read();
  }
}

bool update_goal() {
  if (mode == 5 || mode == 4 || mode == 3) {
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

      
      
      int32_t calculated_sum = vt_1_0 + vt_1_1 + vt_2_0 + vt_2_1 + Vf_0 + Vf_1 + C2_0 + C2_1 + C3_0 + C3_1 + C4_0 + C4_1;
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
      C2[0] = C2_0 / 100000000.0;
      C2[1] = C2_1 / 100000000.0;
      C3[0] = C3_0 / 100000000.0;
      C3[1] = C3_1 / 100000000.0;
      C4[0] = C4_0 / 100000000.0;
      C4[1] = C4_1 / 100000000.0;

      return true;
    }
  }
  else if (mode == 2) {
    if (Serial.available() >= 10) {
      char start_char = Serial.read();
      if (start_char != '\n') {
        while (Serial.available() && Serial.read() != '\n');
        return false;
      }

      uint8_t buffer[8];
      Serial.readBytes(buffer,8);
      if (!Serial.available() || Serial.read() != '\n') {
        return false;
      }

      int32_t xf_0 = *(int32_t*)(buffer + 0);
      int32_t xf_1 = *(int32_t*)(buffer + 4);

      t_init = micros() / 1e6;
      xf[0] = xf_0 / 10000.0;
      xf[1] = xf_1 / 10000.0;

      return true;
    }
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
  dt_sum = dt_angle[0] + dt_angle[1];
  new_xy_vel[0] = (xy_pos[0] - pp_xy_pos[0]) / (dt_sum);
  new_xy_vel[1] =(xy_pos[1] - pp_xy_pos[1]) / (dt_sum);
  new_xy_acc[0] = (dt_angle[0] * pp_xy_pos[0] - dt_sum * p_xy_pos[0] + dt_angle[1]*xy_pos[0])*2/(dt_angle[0]*dt_angle[1]*dt_sum);
  new_xy_acc[1] = (dt_angle[0] * pp_xy_pos[1] - dt_sum * p_xy_pos[1] + dt_angle[1]*xy_pos[1])*2/(dt_angle[0]*dt_angle[1]*dt_sum);

  xy_acc[0] = alpha_acc * new_xy_acc[0] + (1 - alpha_acc) * xy_acc[0];
  xy_acc[1] = alpha_acc * new_xy_acc[1] + (1 - alpha_acc) * xy_acc[1];

  xy_vel[0] = alpha_vel * new_xy_vel[0] + (1 - alpha_vel) * xy_vel[0];
  xy_vel[1] = alpha_vel * new_xy_vel[1] + (1 - alpha_vel) * xy_vel[1];

  auto quant = [&](float x, float xmin, float xmax){
    float y = 2*(x - xmin)/(xmax - xmin) - 1;
    int16_t s = int16_t(round(y * 32767));
    if ((s & 0x000F) == 0x0F) {
        s -= 1;  // Decrease by 1 to make last nibble 0xE
    }
    return s;
  };

  pkt.pos_q[0] = quant(xy_pos[0], -1, 2);  
  pkt.pos_q[1] = quant(xy_pos[1], -1, 2);
  pkt.vel_q[0] = quant(xy_vel[0], -30, 30);
  pkt.vel_q[1] = quant(xy_vel[1], -30, 30);
  pkt.acc_q[0] = quant(xy_acc[0], -150, 150);
  pkt.acc_q[1] = quant(xy_acc[1], -150, 150);

  // checksum over the six int16s (12 bytes)
  pkt.checksum = compute_checksum((uint8_t*)&pkt, sizeof(pkt) - 1);

  Serial.write(0xFF);
  Serial.write(0xFF);
  Serial.write(0xFF);
  Serial.write((uint8_t*)&pkt, sizeof(pkt));
  Serial.write(0x55);
}

void loop() {
  if (mode == 1) {
    read_motor_angles();
    dt = micros() / 1e6 - t_init;
    if (dt > 5e-4) {
      send_motor_pos();
      t_init = micros() / 1e6;
    }
  }
  else {
    update_goal();

    read_motor_angles();
    dt = micros() / 1e6 - t_init;
    apply_voltage(dt);

    send_motor_pos();
  }
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

  if (mode != 4) {
    expected_position(t);
    err[0] = theta_hat[0] - angle[0];
    err[1] = theta_hat[1] - angle[1];
    current_time_err = micros() / 1e6;
    dt_err = current_time_err - past_time_err;
    past_time_err = current_time_err;

    V[0] += kp * err[0];
    V[1] += kp * err[1];

    V[0] += kd * (err[0] - past_err[0]) / dt_err;
    V[1] += kd * (err[1] - past_err[1]) / dt_err;

    accumulated_err[0] += err[0] * dt_err;
    accumulated_err[1] += err[1] * dt_err;

    V[0] += clamp(ki * accumulated_err[0],(float)-5.0,(float)5.0);
    V[1] += clamp(ki * accumulated_err[1],(float)-5.0,(float)5.0);

    past_err[0] = err[0];
    past_err[1] = err[1];
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

void setup_coefficients() {
  double A, B;
  
  for (int i = 0; i < 2; i++) {
    A = sqrt(C6[i] * C6[i] - 4 * C5[i] * C7[i]);
    B = 2 * C7[i] * C7[i] * A;

    ab[i][0] = (-C6[i] - A) / (2 * C5[i]);
    ab[i][1] = (-C6[i] + A) / (2 * C5[i]);

    CE[i][0] = (-C6[i] * C6[i] + A * C6[i] + 2 * C5[i] * C7[i]) / B;
    CE[i][1] = (C6[i] * C6[i] + A * C6[i] - 2 * C5[i] * C7[i]) / B;
    CE[i][2] = 1 / C7[i];
    CE[i][3] = -C6[i] / (C7[i] * C7[i]);

    B = 2 * C7[i] * A;
    A2CE[i][0] = -(-C6[i] + A) / B;
    A2CE[i][1] = -(C6[i] + A) / B;
    A2CE[i][2] = 1 / C7[i];

    A3CE[i][0] = -1 / A;
    A3CE[i][1] = 1 / A;

    B = 2 * C5[i] * A;
    A4CE[i][0] = (C6[i] + A) / B;
    A4CE[i][1] = (-C6[i] + A) / B;
  }
}

double f(double x, int i) {
  return CE[i][0] * eat + CE[i][1] * ebt + CE[i][2] * x + CE[i][3];
}

double g(double tms, int i) {
  if (tms < 0) {
    return 0;
  }
  eat = exp(ab[i][0] * tms);
  ebt = exp(ab[i][1] * tms);
  return f(tms, i);
}

void expected_position(double t) {
  if (mode == 2) {
    xy_to_theta(xf);
  }
  if (t < 0.002) {
    for  (int i = 0; i < 2; i++) {
      expected_xy_pos[i] = C2[i] * (A2CE[i][0] + A2CE[i][1] + A2CE[i][2]) \
                          + C3[i] * (A3CE[i][0] + A3CE[i][1]) \
                          + C4[i] * (A4CE[i][0] + A4CE[i][1]);
    }
  }
  else {
    for (int i = 0; i < 2; i++) {
      eat = exp(ab[i][0] * t);
      ebt = exp(ab[i][1] * t);
      
      A_2 = A2CE[i][0] * eat + A2CE[i][1] * ebt + A2CE[i][2];
      A_3 = A3CE[i][0] * eat + A3CE[i][1] * ebt;
      A_4 = A4CE[i][0] * eat + A4CE[i][1] * ebt;
      
      f_t = f(t, i);
      g_a1 = g(t - vt_1[i], i);
      g_a2 = g(t - vt_2[i], i);
      
      expected_xy_pos[i] = 0.5 * Vf[i] * PULLEY_RADIUS * (f_t - 2 * g_a1 + g_a2) + C2[i] * A_2 + C3[i] * A_3 + C4[i] * A_4;
    }
  }

  xy_to_theta(expected_xy_pos);
}

void read_motor_angles() {
  pp_angle = p_angle;
  p_angle = angle;

  p_encoder = encoder;

  dt_angle[1] = dt_angle[0];

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

  pp_xy_pos = p_xy_pos;
  p_xy_pos = xy_pos;
  theta_to_xy();

  float current_time = micros() / 1e6;
  dt_angle[0] = current_time - past_time_angle;
  past_time_angle = current_time;
}

void theta_to_xy() {
  xy_pos[0] = (angle[0] - angle[1]) * PULLEY_RADIUS * PI / 360;
  xy_pos[1] = (-angle[0] - angle[1]) * PULLEY_RADIUS * PI / 360;
}

void xy_to_theta(array<float,2> xy) {
  theta_hat[0] = (xy[0] - xy[1]) * 360 / (2*PI*PULLEY_RADIUS);
  theta_hat[1] = (-xy[0] - xy[1]) * 360 / (2*PI*PULLEY_RADIUS);
}

void set_motor_pwms(float left, float right) {
  if (left >= 0) {
      digitalWrite(LEFT_MOTOR_DIR_PIN, LOW);
  } else {
      digitalWrite(LEFT_MOTOR_DIR_PIN, HIGH);
  }
  pwm_start(LEFT_MOTOR_PWM_PIN, PWM_FREQ_HZ, floor(abs(left) * DUTY_CYCLE_CONVERSION), RESOLUTION_10B_COMPARE_FORMAT);

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

  //empty_buffer();

  //while (Serial.available() == 0) {
  //  read_motor_angles();
  //}
  //read_motor_angles();
  //left_point = angle;

  //Serial.println("confirmation of left");

  //PULLEY_RADIUS = (360/PI) * (width - 2*(MALLET_RADIUS + wall[1])) / (right_point[0] + right_point[1] - left_point[0] - left_point[1]);
  offset[1] = (360 * (2*MALLET_RADIUS+wall[0]+wall[1]) / (PULLEY_RADIUS * PI) + 2*right_point[1])/(-2);
  offset[0] = (360*(wall[0] - wall[1])/(PULLEY_RADIUS*PI) - 2*right_point[0])/2;

  empty_buffer();
}


/*
//SERIAL DELAY
void setup() {
    Serial.begin(460800);
    while (!Serial);  // Wait for USB Serial to connect
}

void loop() {
    if (Serial.available()) {
        char c = Serial.read();
        if (c == 'P') {
            Serial.write('Q');  // Respond
        }
    }
}
*/

/*
//CAMERA DELAY

#include <Arduino.h>

#define OUTPUT_PIN PB12  // You can change this to any digital output
int delay_offset = 0;

void setup() {
    pinMode(OUTPUT_PIN, OUTPUT);
    digitalWrite(OUTPUT_PIN, LOW);
    Serial.begin(460800);
    while (!Serial);  // Wait for USB CDC serial
}

void loop() {
    // Wait some delay before starting
    delay(100 + ((delay_offset%100) / 10.0));

    // Start timing and set pin high
    unsigned long start_time = micros();
    digitalWrite(OUTPUT_PIN, HIGH);

    // Wait for 'P' from laptop
    while (!Serial.available()) {}
    if (Serial.read() != 'P') return;

    // Stop timing
    unsigned long end_time = micros();
    digitalWrite(OUTPUT_PIN, LOW);

    // Compute elapsed time in milliseconds
    float elapsed_ms = (end_time - start_time);

    // Send float as raw 4-byte binary
    Serial.write((uint8_t*)&elapsed_ms, sizeof(elapsed_ms));
    delay_offset += 1;
}
*/

//Measure total image delay
/*
#include <SPI.h>
#include <cmath>

#define ENC_CHIP_SELECT_LEFT PB12
#define ENC_CHIP_SELECT_RIGHT PA4

#define LEFT_MOTOR_PWM_PIN PB_6
#define RIGHT_MOTOR_PWM_PIN PB_7
#define LEFT_MOTOR_DIR_PIN PB5
#define RIGHT_MOTOR_DIR_PIN PB4
#define LED_PIN PB11  // You can change this to any digital output

#define DUTY_CYCLE_CONVERSION 1024 // Accepted duty cycle values are 0-1024
#define PWM_FREQ_HZ 10000
#define ROLLOVER_ANGLE_DEGS 180
#define MALLET_RADIUS 0.101553/2

using namespace std;

int mode = 5; //1: homing only, 2: feedback only FP, 3: feedback only path, 4: feedforward only, 5: feedback + feedforward
float PULLEY_RADIUS = 0.035306; //meters 

array<float,2> revolutions = {0,0};
array<float,2> offset = {0,0};

array<float,2> angle = {0,0};
array<float,2> p_angle = {0,0};
array<float,2> pp_angle = {0,0};

array<float,2> p_encoder = {0,0};
array<float,2> encoder = {0,0};

array<float,2> xy_pos = {0,0};
array<float,2> p_xy_pos = {0,0};
array<float,2> pp_xy_pos = {0,0};

array<float,2> xy_acc = {0,0};
array<float,2> xy_vel = {0,0};

array<float,2> xf = {0.5,0.5};
array<float,2> expected_xy_pos = {0.4, 0.52};

array<float,2> dt_angle = {0.1,0.1};
float past_time_angle = 0;

array<float,2> past_err = {0,0};
float past_time_err = 0;

array<float,2> theta_hat = {0,0};
array<float,2> err = {0, 0};

float current_time_err = 0;
float dt_err = 0;

array<float,2> accumulated_err = {0,0};

float height = 1.9885;
float width = 0.9905;

float kp = 0.1;
float ki = 0.001;
float kd = 0.005;

array<float,2> wall = {0.0799, 0.08}; // thickness in x and y, mallet R 0.101553/2

float V_max = 20;

const double a1 = 3.579e-6;
const double a2 = 0.00571;
const double a3 = (0.0596 + 0.0467) / 2.0;
const double b1 = -1.7165e-6;
const double b2 = -0.002739;
const double b3 = 0.0;

double C5[2] = {a1 - b1, a1 + b1};
double C6[2] = {a2 - b2, a2 + b2};
double C7[2] = {a3 - b3, a3 + b3};

const double pullyR = 0.035306;

double CE[2][6] = {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}};
double ab[2][2] = {{0, 0}, {0, 0}};
double A2CE[2][3] = {{0, 0, 0}, {0, 0, 0}};
double A3CE[2][2] = {{0, 0}, {0, 0}};
double A4CE[2][2] = {{0, 0}, {0, 0}};

double vt_1[2], vt_2[2], Vf[2], C2[2], C3[2], C4[2];

double t_init = 0;
double dt = 0;

float dt_sum = 0;
array<float,2> new_xy_vel = {0,0};
array<float,2> new_xy_acc = {0,0};

float alpha_acc = 0.05;  // Adjust between 0.0 (no change) and 1.0 (instant change)
float alpha_vel = 0.5;

struct __attribute__((packed)) Packet16 {
  int16_t pos_q[2];
  int16_t vel_q[2];
  int16_t acc_q[2];
  uint8_t checksum;
};

Packet16 pkt;

array<float,2> Vxy = {0,0};
array<float,2> V = {0,0};

double eat, ebt, A_2, A_3, A_4, f_t, g_a1, g_a2;

double timer1 = 0;
double timer2 = 0;
double timer3 = 0;
bool light_on = false;
double camera_dt = 0;
double past_camera_dt = 0;
bool send_time = false;
float ran_delay = 0.03;

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

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);

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

  setup_coefficients();

  t_init = micros() / 1e6;
  vt_1[0] = 0.004;
  vt_1[1] = 0.005;
  vt_2[0] = 0.009;
  vt_2[1] = 0.006;
  Vf[0] = 13.8;
  Vf[1] = 1.8;
  C2[0] = 0.0008;
  C2[1] = 0.0003;
  C3[0] = 0.000009;
  C3[1] = 0.00007987;
  C4[0] = 0.0007654;
  C4[1] = 0.00002459;

  timer1 = t_init;
  timer2 = t_init;
  timer3 = t_init;

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
  t_init = micros() / 1e6;
  timer1 = t_init;

  past_time_angle = micros() / 1e6;
  past_time_err = micros() / 1e6;
}

void empty_buffer() {
  while (Serial.available()) {
    Serial.read();
  }
}

bool update_goal() {
  if (mode == 5 || mode == 4 || mode == 3) {
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

      
      
      int32_t calculated_sum = vt_1_0 + vt_1_1 + vt_2_0 + vt_2_1 + Vf_0 + Vf_1 + C2_0 + C2_1 + C3_0 + C3_1 + C4_0 + C4_1;
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
      C2[0] = C2_0 / 100000000.0;
      C2[1] = C2_1 / 100000000.0;
      C3[0] = C3_0 / 100000000.0;
      C3[1] = C3_1 / 100000000.0;
      C4[0] = C4_0 / 100000000.0;
      C4[1] = C4_1 / 100000000.0;

      if (C4_0 == 75345433) {
        camera_dt = micros() / 1e6 - timer1;
        timer1 = micros() / 1e6;
        //turn LED off
        digitalWrite(LED_PIN, LOW);
        send_time = true;
        light_on = false;
        C4[0] = 324 / 100000000.0;
      }
      else if (C4_0 == 75345432) {
        camera_dt = micros() / 1e6 - timer1;
        timer1 = micros() / 1e6;
        //turn LED off
        digitalWrite(LED_PIN, LOW);
        light_on = false;
        C4[0] = 324 / 100000000.0;
      }


      return true;
    }
  }
  else if (mode == 2) {
    if (Serial.available() >= 10) {
      char start_char = Serial.read();
      if (start_char != '\n') {
        while (Serial.available() && Serial.read() != '\n');
        return false;
      }

      uint8_t buffer[8];
      Serial.readBytes(buffer,8);
      if (!Serial.available() || Serial.read() != '\n') {
        return false;
      }

      int32_t xf_0 = *(int32_t*)(buffer + 0);
      int32_t xf_1 = *(int32_t*)(buffer + 4);

      t_init = micros() / 1e6;
      xf[0] = xf_0 / 10000.0;
      xf[1] = xf_1 / 10000.0;

      return true;
    }
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
  dt_sum = dt_angle[0] + dt_angle[1];
  new_xy_vel[0] = (xy_pos[0] - pp_xy_pos[0]) / (dt_sum);
  new_xy_vel[1] =(xy_pos[1] - pp_xy_pos[1]) / (dt_sum);
  new_xy_acc[0] = (dt_angle[0] * pp_xy_pos[0] - dt_sum * p_xy_pos[0] + dt_angle[1]*xy_pos[0])*2/(dt_angle[0]*dt_angle[1]*dt_sum);
  new_xy_acc[1] = (dt_angle[0] * pp_xy_pos[1] - dt_sum * p_xy_pos[1] + dt_angle[1]*xy_pos[1])*2/(dt_angle[0]*dt_angle[1]*dt_sum);

  xy_acc[0] = alpha_acc * new_xy_acc[0] + (1 - alpha_acc) * xy_acc[0];
  xy_acc[1] = alpha_acc * new_xy_acc[1] + (1 - alpha_acc) * xy_acc[1];

  xy_vel[0] = alpha_vel * new_xy_vel[0] + (1 - alpha_vel) * xy_vel[0];
  xy_vel[1] = alpha_vel * new_xy_vel[1] + (1 - alpha_vel) * xy_vel[1];

  auto quant = [&](float x, float xmin, float xmax){
    float y = 2*(x - xmin)/(xmax - xmin) - 1;
    int16_t s = int16_t(round(y * 32767));
    if ((s & 0x000F) == 0x0F) {
        s -= 1;  // Decrease by 1 to make last nibble 0xE
    }
    return s;
  };

  if (send_time) {
    pkt.pos_q[0] = quant(camera_dt * 1e3, 0, 30);
    past_camera_dt = camera_dt;
    send_time = false;
  } else {
    pkt.pos_q[0] = quant(past_camera_dt * 1e3, 0, 30);
  }
  
  pkt.pos_q[1] = quant(xy_pos[1], -1, 2);
  pkt.vel_q[0] = quant(xy_vel[0], -30, 30);
  pkt.vel_q[1] = quant(xy_vel[1], -30, 30);
  pkt.acc_q[0] = quant(xy_acc[0], -150, 150);
  pkt.acc_q[1] = quant(xy_acc[1], -150, 150);

  // checksum over the six int16s (12 bytes)
  pkt.checksum = compute_checksum((uint8_t*)&pkt, sizeof(pkt) - 1);

  Serial.write(0xFF);
  Serial.write(0xFF);
  Serial.write(0xFF);
  Serial.write((uint8_t*)&pkt, sizeof(pkt));
  Serial.write(0x55);
}

void loop() {
  if (mode == 1) {
    read_motor_angles();
    dt = micros() / 1e6 - t_init;
    if (dt > 5e-4) {
      send_motor_pos();
      t_init = micros() / 1e6;
    }
  }
  else {
    update_goal();

    read_motor_angles();
    dt = micros() / 1e6 - t_init;
    apply_voltage(dt);

    send_motor_pos();
  }

  if (!light_on && (micros() / 1e6 - timer1 > ran_delay)) {
    timer1 = micros() / 1e6;
    counter += 1;
    //turn on LED
    digitalWrite(LED_PIN, HIGH);
    light_on = true;

    long minInt = 0.02 * 10000;
    long maxInt = 0.2 * 10000;

    long r = random(minInt, maxInt + 1);  // +1 to make max inclusive
    ran_delay = r / (float)10000;
  }
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

  if (mode != 4) {
    expected_position(t);
    err[0] = theta_hat[0] - angle[0];
    err[1] = theta_hat[1] - angle[1];
    current_time_err = micros() / 1e6;
    dt_err = current_time_err - past_time_err;
    past_time_err = current_time_err;

    V[0] += kp * err[0];
    V[1] += kp * err[1];

    V[0] += kd * (err[0] - past_err[0]) / dt_err;
    V[1] += kd * (err[1] - past_err[1]) / dt_err;

    accumulated_err[0] += err[0] * dt_err;
    accumulated_err[1] += err[1] * dt_err;

    V[0] += clamp(ki * accumulated_err[0],(float)-5.0,(float)5.0);
    V[1] += clamp(ki * accumulated_err[1],(float)-5.0,(float)5.0);

    past_err[0] = err[0];
    past_err[1] = err[1];
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

void setup_coefficients() {
  double A, B;
  
  for (int i = 0; i < 2; i++) {
    A = sqrt(C6[i] * C6[i] - 4 * C5[i] * C7[i]);
    B = 2 * C7[i] * C7[i] * A;

    ab[i][0] = (-C6[i] - A) / (2 * C5[i]);
    ab[i][1] = (-C6[i] + A) / (2 * C5[i]);

    CE[i][0] = (-C6[i] * C6[i] + A * C6[i] + 2 * C5[i] * C7[i]) / B;
    CE[i][1] = (C6[i] * C6[i] + A * C6[i] - 2 * C5[i] * C7[i]) / B;
    CE[i][2] = 1 / C7[i];
    CE[i][3] = -C6[i] / (C7[i] * C7[i]);

    B = 2 * C7[i] * A;
    A2CE[i][0] = -(-C6[i] + A) / B;
    A2CE[i][1] = -(C6[i] + A) / B;
    A2CE[i][2] = 1 / C7[i];

    A3CE[i][0] = -1 / A;
    A3CE[i][1] = 1 / A;

    B = 2 * C5[i] * A;
    A4CE[i][0] = (C6[i] + A) / B;
    A4CE[i][1] = (-C6[i] + A) / B;
  }
}

double f(double x, int i) {
  return CE[i][0] * eat + CE[i][1] * ebt + CE[i][2] * x + CE[i][3];
}

double g(double tms, int i) {
  if (tms < 0) {
    return 0;
  }
  eat = exp(ab[i][0] * tms);
  ebt = exp(ab[i][1] * tms);
  return f(tms, i);
}

void expected_position(double t) {
  if (mode == 2) {
    xy_to_theta(xf);
  }
  if (t < 0.002) {
    for  (int i = 0; i < 2; i++) {
      expected_xy_pos[i] = C2[i] * (A2CE[i][0] + A2CE[i][1] + A2CE[i][2]) \
                          + C3[i] * (A3CE[i][0] + A3CE[i][1]) \
                          + C4[i] * (A4CE[i][0] + A4CE[i][1]);
    }
  }
  else {
    for (int i = 0; i < 2; i++) {
      eat = exp(ab[i][0] * t);
      ebt = exp(ab[i][1] * t);
      
      A_2 = A2CE[i][0] * eat + A2CE[i][1] * ebt + A2CE[i][2];
      A_3 = A3CE[i][0] * eat + A3CE[i][1] * ebt;
      A_4 = A4CE[i][0] * eat + A4CE[i][1] * ebt;
      
      f_t = f(t, i);
      g_a1 = g(t - vt_1[i], i);
      g_a2 = g(t - vt_2[i], i);
      
      expected_xy_pos[i] = 0.5 * Vf[i] * pullyR * (f_t - 2 * g_a1 + g_a2) + C2[i] * A_2 + C3[i] * A_3 + C4[i] * A_4;
    }
  }

  xy_to_theta(expected_xy_pos);
}

void read_motor_angles() {
  pp_angle = p_angle;
  p_angle = angle;

  p_encoder = encoder;

  dt_angle[1] = dt_angle[0];

  digitalWrite(chips[0], LOW);
  serial_response = SPI.transfer16(0x3FFF);
  digitalWrite(chips[0], HIGH);
  encoder[0] = (serial_response & 0b0011111111111111) * 360.0 / 16384;

  if(encoder[0] - p_encoder[0] > ROLLOVER_ANGLE_DEGS) {
      revolutions[0] += 1;
  } else if(p_encoder[0] - encoder[0] > ROLLOVER_ANGLE_DEGS) {
      revolutions[0] -= 1;
  }

  digitalWrite(chips[0], LOW);
  serial_response = SPI.transfer16(0x3FFF);
  digitalWrite(chips[0], HIGH);
  encoder[1] = (serial_response & 0b0011111111111111) * 360.0 / 16384;

  if(encoder[1] - p_encoder[1] > ROLLOVER_ANGLE_DEGS) {
      revolutions[1] += 1;
  } else if (p_encoder[1] - encoder[1] > ROLLOVER_ANGLE_DEGS) {
      revolutions[1] -= 1;
  }

  angle[0] = -encoder[0] + 360.0 * revolutions[0] + offset[0];
  angle[1] = -encoder[1] + 360.0 * revolutions[1] + offset[1];

  pp_xy_pos = p_xy_pos;
  p_xy_pos = xy_pos;
  theta_to_xy();

  float current_time = micros() / 1e6;
  dt_angle[0] = current_time - past_time_angle;
  past_time_angle = current_time;
}

void theta_to_xy() {
  xy_pos[0] = (angle[0] - angle[1]) * PULLEY_RADIUS * PI / 360;
  xy_pos[1] = (-angle[0] - angle[1]) * PULLEY_RADIUS * PI / 360;
}

void xy_to_theta(array<float,2> xy) {
  theta_hat[0] = (xy[0] - xy[1]) * 360 / (2*PI*PULLEY_RADIUS);
  theta_hat[1] = (-xy[0] - xy[1]) * 360 / (2*PI*PULLEY_RADIUS);
}

void set_motor_pwms(float left, float right) {
  if (left >= 0) {
      digitalWrite(LEFT_MOTOR_DIR_PIN, LOW);
  } else {
      digitalWrite(LEFT_MOTOR_DIR_PIN, HIGH);
  }
  pwm_start(LEFT_MOTOR_PWM_PIN, PWM_FREQ_HZ, floor(abs(left) * DUTY_CYCLE_CONVERSION), RESOLUTION_10B_COMPARE_FORMAT);

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

  Serial.println("confirmation of left");

  PULLEY_RADIUS = (360/PI) * (width - 2*(MALLET_RADIUS + wall[1])) / (right_point[0] + right_point[1] - left_point[0] - left_point[1]);
  offset[1] = (360 * (2*MALLET_RADIUS+wall[0]+wall[1]) / (PULLEY_RADIUS * PI) + 2*right_point[1])/(-2);
  offset[0] = (360*(wall[0] - wall[1])/(PULLEY_RADIUS*PI) - 2*right_point[0])/2;
}
*/

//Measures total mallet delay
/*
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
#define MALLET_RADIUS 0.101553/2

using namespace std;

int mode = 5; //1: homing only, 2: feedback only FP, 3: feedback only path, 4: feedforward only, 5: feedback + feedforward
float PULLEY_RADIUS = 0.035306; //meters 

array<float,2> revolutions = {0,0};
array<float,2> offset = {0,0};

array<float,2> angle = {0,0};
array<float,2> p_angle = {0,0};
array<float,2> pp_angle = {0,0};

array<float,2> p_encoder = {0,0};
array<float,2> encoder = {0,0};

array<float,2> xy_pos = {0,0};
array<float,2> p_xy_pos = {0,0};
array<float,2> pp_xy_pos = {0,0};

array<float,2> xy_acc = {0,0};
array<float,2> xy_vel = {0,0};

array<float,2> xf = {0.5,0.5};
array<float,2> expected_xy_pos = {0.4, 0.52};

array<float,2> dt_angle = {0.1,0.1};
float past_time_angle = 0;

array<float,2> past_err = {0,0};
float past_time_err = 0;

array<float,2> theta_hat = {0,0};
array<float,2> err = {0, 0};

float current_time_err = 0;
float dt_err = 0;

array<float,2> accumulated_err = {0,0};

float height = 1.9885;
float width = 0.9905;

float kp = 0.1;
float ki = 0.001;
float kd = 0.005;

array<float,2> wall = {0.0799, 0.08}; // thickness in x and y, mallet R 0.101553/2

float V_max = 20;

const double a1 = 3.579e-6;
const double a2 = 0.00571;
const double a3 = (0.0596 + 0.0467) / 2.0;
const double b1 = -1.7165e-6;
const double b2 = -0.002739;
const double b3 = 0.0;

double C5[2] = {a1 - b1, a1 + b1};
double C6[2] = {a2 - b2, a2 + b2};
double C7[2] = {a3 - b3, a3 + b3};

const double pullyR = 0.035306;

double CE[2][6] = {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}};
double ab[2][2] = {{0, 0}, {0, 0}};
double A2CE[2][3] = {{0, 0, 0}, {0, 0, 0}};
double A3CE[2][2] = {{0, 0}, {0, 0}};
double A4CE[2][2] = {{0, 0}, {0, 0}};

double vt_1[2], vt_2[2], Vf[2], C2[2], C3[2], C4[2];

double t_init = 0;
double dt = 0;

float dt_sum = 0;
array<float,2> new_xy_vel = {0,0};
array<float,2> new_xy_acc = {0,0};

float alpha_acc = 0.05;  // Adjust between 0.0 (no change) and 1.0 (instant change)
float alpha_vel = 0.5;

struct __attribute__((packed)) Packet16 {
  int16_t pos_q[2];
  int16_t vel_q[2];
  int16_t acc_q[2];
  uint8_t checksum;
};

Packet16 pkt;

array<float,2> Vxy = {0,0};
array<float,2> V = {0,0};

double eat, ebt, A_2, A_3, A_4, f_t, g_a1, g_a2;

double timer1 = 0;
double mallet_delay = 0;
double read_encoder_time = 0;

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

  setup_coefficients();

  t_init = micros() / 1e6;
  vt_1[0] = 0.004;
  vt_1[1] = 0.005;
  vt_2[0] = 0.009;
  vt_2[1] = 0.006;
  Vf[0] = 13.8;
  Vf[1] = 1.8;
  C2[0] = 0.0008;
  C2[1] = 0.0003;
  C3[0] = 0.000009;
  C3[1] = 0.00007987;
  C4[0] = 0.0007654;
  C4[1] = 0.00002459;

  timer1 = t_init;

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
  t_init = micros() / 1e6;
  timer1 = t_init;

  past_time_angle = micros() / 1e6;
  past_time_err = micros() / 1e6;
}

void empty_buffer() {
  while (Serial.available()) {
    Serial.read();
  }
}

bool update_goal() {
  if (mode == 5 || mode == 4 || mode == 3) {
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
      
      int32_t calculated_sum = vt_1_0 + vt_1_1 + vt_2_0 + vt_2_1 + Vf_0 + Vf_1 + C2_0 + C2_1 + C3_0 + C3_1 + C4_0 + C4_1;
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
      C2[0] = C2_0 / 100000000.0;
      C2[1] = C2_1 / 100000000.0;
      C3[0] = C3_0 / 100000000.0;
      C3[1] = C3_1 / 100000000.0;
      // C4[0] = C4_0 / 100000000.0;
      C4[1] = C4_1 / 100000000.0;

      mallet_delay = (micros() / 1e6 - timer1) - (C4_0 / 100000.0) / 1000.0;
      C4[0] = 300 / 100000000.0;
      timer1 = micros() / 1e6;

      return true;
    }
  }
  else if (mode == 2) {
    if (Serial.available() >= 10) {
      char start_char = Serial.read();
      if (start_char != '\n') {
        while (Serial.available() && Serial.read() != '\n');
        return false;
      }

      uint8_t buffer[8];
      Serial.readBytes(buffer,8);
      if (!Serial.available() || Serial.read() != '\n') {
        return false;
      }

      int32_t xf_0 = *(int32_t*)(buffer + 0);
      int32_t xf_1 = *(int32_t*)(buffer + 4);

      t_init = micros() / 1e6;
      xf[0] = xf_0 / 10000.0;
      xf[1] = xf_1 / 10000.0;

      return true;
    }
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
  dt_sum = dt_angle[0] + dt_angle[1];
  new_xy_vel[0] = (xy_pos[0] - pp_xy_pos[0]) / (dt_sum);
  new_xy_vel[1] =(xy_pos[1] - pp_xy_pos[1]) / (dt_sum);
  new_xy_acc[0] = (dt_angle[0] * pp_xy_pos[0] - dt_sum * p_xy_pos[0] + dt_angle[1]*xy_pos[0])*2/(dt_angle[0]*dt_angle[1]*dt_sum);
  new_xy_acc[1] = (dt_angle[0] * pp_xy_pos[1] - dt_sum * p_xy_pos[1] + dt_angle[1]*xy_pos[1])*2/(dt_angle[0]*dt_angle[1]*dt_sum);

  xy_acc[0] = alpha_acc * new_xy_acc[0] + (1 - alpha_acc) * xy_acc[0];
  xy_acc[1] = alpha_acc * new_xy_acc[1] + (1 - alpha_acc) * xy_acc[1];

  xy_vel[0] = alpha_vel * new_xy_vel[0] + (1 - alpha_vel) * xy_vel[0];
  xy_vel[1] = alpha_vel * new_xy_vel[1] + (1 - alpha_vel) * xy_vel[1];

  auto quant = [&](float x, float xmin, float xmax){
    float y = 2*(x - xmin)/(xmax - xmin) - 1;
    int16_t s = int16_t(round(y * 32767));
    if ((s & 0x000F) == 0x0F) {
        s -= 1;  // Decrease by 1 to make last nibble 0xE
    }
    return s;
  };

  pkt.pos_q[0] = quant(xy_pos[0], -1, 2);
  pkt.pos_q[1] = quant(xy_pos[1], -1, 2);
  pkt.vel_q[0] = quant((read_encoder_time - timer1) * 1e3, 0, 40);
  pkt.vel_q[1] = quant(mallet_delay * 1e3, 0, 40);
  //pkt.vel_q[0] = quant(xy_vel[0], -30, 30);
  //pkt.vel_q[1] = quant(xy_vel[1], -30, 30);
  pkt.acc_q[0] = quant(xy_acc[0], -150, 150);
  pkt.acc_q[1] = quant(xy_acc[1], -150, 150);

  // checksum over the six int16s (12 bytes)
  pkt.checksum = compute_checksum((uint8_t*)&pkt, sizeof(pkt) - 1);

  Serial.write(0xFF);
  Serial.write(0xFF);
  Serial.write(0xFF);
  Serial.write((uint8_t*)&pkt, sizeof(pkt));
  Serial.write(0x55);
}

void loop() {
  if (mode == 1) {
    read_motor_angles();
    dt = micros() / 1e6 - t_init;
    if (dt > 5e-4) {
      send_motor_pos();
      t_init = micros() / 1e6;
    }
  }
  else {
    update_goal();

    read_motor_angles();
    dt = micros() / 1e6 - t_init;
    apply_voltage(dt);

    send_motor_pos();
  }
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

  if (mode != 4) {
    expected_position(t);
    err[0] = theta_hat[0] - angle[0];
    err[1] = theta_hat[1] - angle[1];
    current_time_err = micros() / 1e6;
    dt_err = current_time_err - past_time_err;
    past_time_err = current_time_err;

    V[0] += kp * err[0];
    V[1] += kp * err[1];

    V[0] += kd * (err[0] - past_err[0]) / dt_err;
    V[1] += kd * (err[1] - past_err[1]) / dt_err;

    accumulated_err[0] += err[0] * dt_err;
    accumulated_err[1] += err[1] * dt_err;

    V[0] += clamp(ki * accumulated_err[0],(float)-5.0,(float)5.0);
    V[1] += clamp(ki * accumulated_err[1],(float)-5.0,(float)5.0);

    past_err[0] = err[0];
    past_err[1] = err[1];
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

void setup_coefficients() {
  double A, B;
  
  for (int i = 0; i < 2; i++) {
    A = sqrt(C6[i] * C6[i] - 4 * C5[i] * C7[i]);
    B = 2 * C7[i] * C7[i] * A;

    ab[i][0] = (-C6[i] - A) / (2 * C5[i]);
    ab[i][1] = (-C6[i] + A) / (2 * C5[i]);

    CE[i][0] = (-C6[i] * C6[i] + A * C6[i] + 2 * C5[i] * C7[i]) / B;
    CE[i][1] = (C6[i] * C6[i] + A * C6[i] - 2 * C5[i] * C7[i]) / B;
    CE[i][2] = 1 / C7[i];
    CE[i][3] = -C6[i] / (C7[i] * C7[i]);

    B = 2 * C7[i] * A;
    A2CE[i][0] = -(-C6[i] + A) / B;
    A2CE[i][1] = -(C6[i] + A) / B;
    A2CE[i][2] = 1 / C7[i];

    A3CE[i][0] = -1 / A;
    A3CE[i][1] = 1 / A;

    B = 2 * C5[i] * A;
    A4CE[i][0] = (C6[i] + A) / B;
    A4CE[i][1] = (-C6[i] + A) / B;
  }
}

double f(double x, int i) {
  return CE[i][0] * eat + CE[i][1] * ebt + CE[i][2] * x + CE[i][3];
}

double g(double tms, int i) {
  if (tms < 0) {
    return 0;
  }
  eat = exp(ab[i][0] * tms);
  ebt = exp(ab[i][1] * tms);
  return f(tms, i);
}

void expected_position(double t) {
  if (mode == 2) {
    xy_to_theta(xf);
  }
  if (t < 0.002) {
    for  (int i = 0; i < 2; i++) {
      expected_xy_pos[i] = C2[i] * (A2CE[i][0] + A2CE[i][1] + A2CE[i][2]) \
                          + C3[i] * (A3CE[i][0] + A3CE[i][1]) \
                          + C4[i] * (A4CE[i][0] + A4CE[i][1]);
    }
  }
  else {
    for (int i = 0; i < 2; i++) {
      eat = exp(ab[i][0] * t);
      ebt = exp(ab[i][1] * t);
      
      A_2 = A2CE[i][0] * eat + A2CE[i][1] * ebt + A2CE[i][2];
      A_3 = A3CE[i][0] * eat + A3CE[i][1] * ebt;
      A_4 = A4CE[i][0] * eat + A4CE[i][1] * ebt;
      
      f_t = f(t, i);
      g_a1 = g(t - vt_1[i], i);
      g_a2 = g(t - vt_2[i], i);
      
      expected_xy_pos[i] = 0.5 * Vf[i] * pullyR * (f_t - 2 * g_a1 + g_a2) + C2[i] * A_2 + C3[i] * A_3 + C4[i] * A_4;
    }
  }

  xy_to_theta(expected_xy_pos);
}

void read_motor_angles() {
  pp_angle = p_angle;
  p_angle = angle;

  p_encoder = encoder;

  dt_angle[1] = dt_angle[0];

  digitalWrite(chips[0], LOW);
  serial_response = SPI.transfer16(0x3FFF);
  digitalWrite(chips[0], HIGH);
  encoder[0] = (serial_response & 0b0011111111111111) * 360.0 / 16384;

  if(encoder[0] - p_encoder[0] > ROLLOVER_ANGLE_DEGS) {
      revolutions[0] += 1;
  } else if(p_encoder[0] - encoder[0] > ROLLOVER_ANGLE_DEGS) {
      revolutions[0] -= 1;
  }

  digitalWrite(chips[0], LOW);
  serial_response = SPI.transfer16(0x3FFF);
  digitalWrite(chips[0], HIGH);
  encoder[1] = (serial_response & 0b0011111111111111) * 360.0 / 16384;

  read_encoder_time = micros() / 1e6;

  if(encoder[1] - p_encoder[1] > ROLLOVER_ANGLE_DEGS) {
      revolutions[1] += 1;
  } else if (p_encoder[1] - encoder[1] > ROLLOVER_ANGLE_DEGS) {
      revolutions[1] -= 1;
  }

  angle[0] = -encoder[0] + 360.0 * revolutions[0] + offset[0];
  angle[1] = -encoder[1] + 360.0 * revolutions[1] + offset[1];

  pp_xy_pos = p_xy_pos;
  p_xy_pos = xy_pos;
  theta_to_xy();

  float current_time = micros() / 1e6;
  dt_angle[0] = current_time - past_time_angle;
  past_time_angle = current_time;
}

void theta_to_xy() {
  xy_pos[0] = (angle[0] - angle[1]) * PULLEY_RADIUS * PI / 360;
  xy_pos[1] = (-angle[0] - angle[1]) * PULLEY_RADIUS * PI / 360;
}

void xy_to_theta(array<float,2> xy) {
  theta_hat[0] = (xy[0] - xy[1]) * 360 / (2*PI*PULLEY_RADIUS);
  theta_hat[1] = (-xy[0] - xy[1]) * 360 / (2*PI*PULLEY_RADIUS);
}

void set_motor_pwms(float left, float right) {
  if (left >= 0) {
      digitalWrite(LEFT_MOTOR_DIR_PIN, LOW);
  } else {
      digitalWrite(LEFT_MOTOR_DIR_PIN, HIGH);
  }
  pwm_start(LEFT_MOTOR_PWM_PIN, PWM_FREQ_HZ, floor(abs(left) * DUTY_CYCLE_CONVERSION), RESOLUTION_10B_COMPARE_FORMAT);

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

  Serial.println("confirmation of left");

  PULLEY_RADIUS = (360/PI) * (width - 2*(MALLET_RADIUS + wall[1])) / (right_point[0] + right_point[1] - left_point[0] - left_point[1]);
  offset[1] = (360 * (2*MALLET_RADIUS+wall[0]+wall[1]) / (PULLEY_RADIUS * PI) + 2*right_point[1])/(-2);
  offset[0] = (360*(wall[0] - wall[1])/(PULLEY_RADIUS*PI) - 2*right_point[0])/2;
}
*/

//Sends pully radius
/*
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
float PULLEY_RADIUS = 0.035306; //meters 

array<float,2> revolutions = {0,0};
array<float,2> offset = {0,0};

array<float,2> angle = {0,0};
array<float,2> p_angle = {0,0};
array<float,2> pp_angle = {0,0};

array<float,2> p_encoder = {0,0};
array<float,2> encoder = {0,0};

array<float,2> xy_pos = {0,0};
array<float,2> p_xy_pos = {0,0};
array<float,2> pp_xy_pos = {0,0};

array<float,2> xy_acc = {0,0};
array<float,2> xy_vel = {0,0};

array<float,2> xf = {0.5,0.5};
array<float,2> expected_xy_pos = {0.4, 0.52};

array<float,2> dt_angle = {0.1,0.1};
float past_time_angle = 0;

array<float,2> past_err = {0,0};
float past_time_err = 0;

array<float,2> theta_hat = {0,0};
array<float,2> err = {0, 0};

float current_time_err = 0;
float dt_err = 0;

array<float,2> accumulated_err = {0,0};

float height = 1.993;
float width = 0.992;

float kp = 0.1;
float ki = 0.001;
float kd = 0.005;

array<float,2> wall = {0.07974, 0.08022}; // thickness in x and y, mallet R 0.101553/2

float V_max = 20;

const double a1 = 3.579e-6;
const double a2 = 0.00571;
const double a3 = (0.0596 + 0.0467) / 2.0;
const double b1 = -1.7165e-6;
const double b2 = -0.002739;
const double b3 = 0.0;

double C5[2] = {a1 - b1, a1 + b1};
double C6[2] = {a2 - b2, a2 + b2};
double C7[2] = {a3 - b3, a3 + b3};

const double pullyR = 0.035306;

double CE[2][6] = {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}};
double ab[2][2] = {{0, 0}, {0, 0}};
double A2CE[2][3] = {{0, 0, 0}, {0, 0, 0}};
double A3CE[2][2] = {{0, 0}, {0, 0}};
double A4CE[2][2] = {{0, 0}, {0, 0}};

double vt_1[2], vt_2[2], Vf[2], C2[2], C3[2], C4[2];

double t_init = 0;
double dt = 0;

float dt_sum = 0;
array<float,2> new_xy_vel = {0,0};
array<float,2> new_xy_acc = {0,0};

float alpha_acc = 0.05;  // Adjust between 0.0 (no change) and 1.0 (instant change)
float alpha_vel = 0.5;

struct __attribute__((packed)) Packet16 {
  int16_t pos_q[2];
  int16_t vel_q[2];
  int16_t acc_q[2];
  uint8_t checksum;
};

Packet16 pkt;

array<float,2> Vxy = {0,0};
array<float,2> V = {0,0};

double eat, ebt, A_2, A_3, A_4, f_t, g_a1, g_a2;

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

  setup_coefficients();

  t_init = micros() / 1e6;
  vt_1[0] = 0.004;
  vt_1[1] = 0.005;
  vt_2[0] = 0.009;
  vt_2[1] = 0.006;
  Vf[0] = 13.8;
  Vf[1] = 1.8;
  C2[0] = 0.0008;
  C2[1] = 0.0003;
  C3[0] = 0.000009;
  C3[1] = 0.00007987;
  C4[0] = 0.0007654;
  C4[1] = 0.00002459;

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
  home_table();

  t_init = micros() / 1e6;

  past_time_angle = micros() / 1e6;
  past_time_err = micros() / 1e6;
}

void empty_buffer() {
  while (Serial.available()) {
    Serial.read();
  }
}

bool update_goal() {
  if (mode == 5 || mode == 4 || mode == 3) {
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

      
      
      int32_t calculated_sum = vt_1_0 + vt_1_1 + vt_2_0 + vt_2_1 + Vf_0 + Vf_1 + C2_0 + C2_1 + C3_0 + C3_1 + C4_0 + C4_1;
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
      C2[0] = C2_0 / 100000000.0;
      C2[1] = C2_1 / 100000000.0;
      C3[0] = C3_0 / 100000000.0;
      C3[1] = C3_1 / 100000000.0;
      C4[0] = C4_0 / 100000000.0;
      C4[1] = C4_1 / 100000000.0;

      return true;
    }
  }
  else if (mode == 2) {
    if (Serial.available() >= 10) {
      char start_char = Serial.read();
      if (start_char != '\n') {
        while (Serial.available() && Serial.read() != '\n');
        return false;
      }

      uint8_t buffer[8];
      Serial.readBytes(buffer,8);
      if (!Serial.available() || Serial.read() != '\n') {
        return false;
      }

      int32_t xf_0 = *(int32_t*)(buffer + 0);
      int32_t xf_1 = *(int32_t*)(buffer + 4);

      t_init = micros() / 1e6;
      xf[0] = xf_0 / 10000.0;
      xf[1] = xf_1 / 10000.0;

      return true;
    }
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
  dt_sum = dt_angle[0] + dt_angle[1];
  new_xy_vel[0] = (xy_pos[0] - pp_xy_pos[0]) / (dt_sum);
  new_xy_vel[1] =(xy_pos[1] - pp_xy_pos[1]) / (dt_sum);
  new_xy_acc[0] = (dt_angle[0] * pp_xy_pos[0] - dt_sum * p_xy_pos[0] + dt_angle[1]*xy_pos[0])*2/(dt_angle[0]*dt_angle[1]*dt_sum);
  new_xy_acc[1] = (dt_angle[0] * pp_xy_pos[1] - dt_sum * p_xy_pos[1] + dt_angle[1]*xy_pos[1])*2/(dt_angle[0]*dt_angle[1]*dt_sum);

  xy_acc[0] = alpha_acc * new_xy_acc[0] + (1 - alpha_acc) * xy_acc[0];
  xy_acc[1] = alpha_acc * new_xy_acc[1] + (1 - alpha_acc) * xy_acc[1];

  xy_vel[0] = alpha_vel * new_xy_vel[0] + (1 - alpha_vel) * xy_vel[0];
  xy_vel[1] = alpha_vel * new_xy_vel[1] + (1 - alpha_vel) * xy_vel[1];

  auto quant = [&](float x, float xmin, float xmax){
    float y = 2*(x - xmin)/(xmax - xmin) - 1;
    int16_t s = int16_t(round(y * 32767));
    if ((s & 0x000F) == 0x0F) {
        s -= 1;  // Decrease by 1 to make last nibble 0xE
    }
    return s;
  };

  pkt.pos_q[0] = quant(PULLEY_RADIUS, 0.03, 0.04);
  //pkt.pos_q[0] = quant(xy_pos[0], -1, 2);  
  pkt.pos_q[1] = quant(xy_pos[1], -1, 2);
  pkt.vel_q[0] = quant(xy_vel[0], -30, 30);
  pkt.vel_q[1] = quant(xy_vel[1], -30, 30);
  pkt.acc_q[0] = quant(xy_acc[0], -150, 150);
  pkt.acc_q[1] = quant(xy_acc[1], -150, 150);

  // checksum over the six int16s (12 bytes)
  pkt.checksum = compute_checksum((uint8_t*)&pkt, sizeof(pkt) - 1);

  Serial.write(0xFF);
  Serial.write(0xFF);
  Serial.write(0xFF);
  Serial.write((uint8_t*)&pkt, sizeof(pkt));
  Serial.write(0x55);
}

void loop() {
  if (mode == 1) {
    read_motor_angles();
    dt = micros() / 1e6 - t_init;
    if (dt > 5e-4) {
      send_motor_pos();
      t_init = micros() / 1e6;
    }
  }
  else {
    update_goal();

    read_motor_angles();
    dt = micros() / 1e6 - t_init;
    apply_voltage(dt);

    send_motor_pos();
  }
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

  if (mode != 4) {
    expected_position(t);
    err[0] = theta_hat[0] - angle[0];
    err[1] = theta_hat[1] - angle[1];
    current_time_err = micros() / 1e6;
    dt_err = current_time_err - past_time_err;
    past_time_err = current_time_err;

    V[0] += kp * err[0];
    V[1] += kp * err[1];

    V[0] += kd * (err[0] - past_err[0]) / dt_err;
    V[1] += kd * (err[1] - past_err[1]) / dt_err;

    accumulated_err[0] += err[0] * dt_err;
    accumulated_err[1] += err[1] * dt_err;

    V[0] += clamp(ki * accumulated_err[0],(float)-5.0,(float)5.0);
    V[1] += clamp(ki * accumulated_err[1],(float)-5.0,(float)5.0);

    past_err[0] = err[0];
    past_err[1] = err[1];
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

void setup_coefficients() {
  double A, B;
  
  for (int i = 0; i < 2; i++) {
    A = sqrt(C6[i] * C6[i] - 4 * C5[i] * C7[i]);
    B = 2 * C7[i] * C7[i] * A;

    ab[i][0] = (-C6[i] - A) / (2 * C5[i]);
    ab[i][1] = (-C6[i] + A) / (2 * C5[i]);

    CE[i][0] = (-C6[i] * C6[i] + A * C6[i] + 2 * C5[i] * C7[i]) / B;
    CE[i][1] = (C6[i] * C6[i] + A * C6[i] - 2 * C5[i] * C7[i]) / B;
    CE[i][2] = 1 / C7[i];
    CE[i][3] = -C6[i] / (C7[i] * C7[i]);

    B = 2 * C7[i] * A;
    A2CE[i][0] = -(-C6[i] + A) / B;
    A2CE[i][1] = -(C6[i] + A) / B;
    A2CE[i][2] = 1 / C7[i];

    A3CE[i][0] = -1 / A;
    A3CE[i][1] = 1 / A;

    B = 2 * C5[i] * A;
    A4CE[i][0] = (C6[i] + A) / B;
    A4CE[i][1] = (-C6[i] + A) / B;
  }
}

double f(double x, int i) {
  return CE[i][0] * eat + CE[i][1] * ebt + CE[i][2] * x + CE[i][3];
}

double g(double tms, int i) {
  if (tms < 0) {
    return 0;
  }
  eat = exp(ab[i][0] * tms);
  ebt = exp(ab[i][1] * tms);
  return f(tms, i);
}

void expected_position(double t) {
  if (mode == 2) {
    xy_to_theta(xf);
  }
  if (t < 0.002) {
    for  (int i = 0; i < 2; i++) {
      expected_xy_pos[i] = C2[i] * (A2CE[i][0] + A2CE[i][1] + A2CE[i][2]) \
                          + C3[i] * (A3CE[i][0] + A3CE[i][1]) \
                          + C4[i] * (A4CE[i][0] + A4CE[i][1]);
    }
  }
  else {
    for (int i = 0; i < 2; i++) {
      eat = exp(ab[i][0] * t);
      ebt = exp(ab[i][1] * t);
      
      A_2 = A2CE[i][0] * eat + A2CE[i][1] * ebt + A2CE[i][2];
      A_3 = A3CE[i][0] * eat + A3CE[i][1] * ebt;
      A_4 = A4CE[i][0] * eat + A4CE[i][1] * ebt;
      
      f_t = f(t, i);
      g_a1 = g(t - vt_1[i], i);
      g_a2 = g(t - vt_2[i], i);
      
      expected_xy_pos[i] = 0.5 * Vf[i] * pullyR * (f_t - 2 * g_a1 + g_a2) + C2[i] * A_2 + C3[i] * A_3 + C4[i] * A_4;
    }
  }

  xy_to_theta(expected_xy_pos);
}

void read_motor_angles() {
  pp_angle = p_angle;
  p_angle = angle;

  p_encoder = encoder;

  dt_angle[1] = dt_angle[0];

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

  pp_xy_pos = p_xy_pos;
  p_xy_pos = xy_pos;
  theta_to_xy();

  float current_time = micros() / 1e6;
  dt_angle[0] = current_time - past_time_angle;
  past_time_angle = current_time;
}

void theta_to_xy() {
  xy_pos[0] = (angle[0] - angle[1]) * PULLEY_RADIUS * PI / 360;
  xy_pos[1] = (-angle[0] - angle[1]) * PULLEY_RADIUS * PI / 360;
}

void xy_to_theta(array<float,2> xy) {
  theta_hat[0] = (xy[0] - xy[1]) * 360 / (2*PI*PULLEY_RADIUS);
  theta_hat[1] = (-xy[0] - xy[1]) * 360 / (2*PI*PULLEY_RADIUS);
}

void set_motor_pwms(float left, float right) {
  if (left >= 0) {
      digitalWrite(LEFT_MOTOR_DIR_PIN, LOW);
  } else {
      digitalWrite(LEFT_MOTOR_DIR_PIN, HIGH);
  }
  pwm_start(LEFT_MOTOR_PWM_PIN, PWM_FREQ_HZ, floor(abs(left) * DUTY_CYCLE_CONVERSION), RESOLUTION_10B_COMPARE_FORMAT);

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

  Serial.println("confirmation of left");

  PULLEY_RADIUS = (360/PI) * (width - 2*(MALLET_RADIUS + wall[1])) / (right_point[0] + right_point[1] - left_point[0] - left_point[1]);
  offset[1] = (360 * (2*MALLET_RADIUS+wall[0]+wall[1]) / (PULLEY_RADIUS * PI) + 2*right_point[1])/(-2);
  offset[0] = (360*(wall[0] - wall[1])/(PULLEY_RADIUS*PI) - 2*right_point[0])/2;

  empty_buffer();
}
*/
