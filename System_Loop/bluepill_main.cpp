#include <Arduino.h>
#include <ArduinoJson.h>
#include <SPI.h>
#include <cmath>

#define ENC_CHIP_SELECT_LEFT PB12
#define ENC_CHIP_SELECT_RIGHT PA4
#define LEFT_MOTOR_PWM_PIN PB_6
#define RIGHT_MOTOR_PWM_PIN PB_7
#define LEFT_MOTOR_DIR_PIN PB3
#define RIGHT_MOTOR_DIR_PIN PB4
#define DUTY_CYCLE_CONVERSION 1024 // Accepted duty cycle values are 0-1024
#define PWM_FREQ_HZ 10000
#define ROLLOVER_ANGLE_DEGS 180
#define PULLEY_RADIUS 0.035306 //meters
#define MALLET_RADIUS 0.08 // change later

using namespace std;

int mode = 1; //1: homing only, 2: feedback only FP, 3: feedback only path, 4: feedforward only, 5: feedback + feedforward

array<float,2> revolutions = {0,0};
array<float,2> offset = {0,0};

array<float,2> angle = {0,0};
array<float,2> p_angle = {0,0};
array<float,2> pp_angle = {0,0};

array<float,2> xy_pos = {0,0};
array<float,2> p_xy_pos = {0,0};
array<float,2> pp_xy_pos = {0,0};

array<float,2> xf = {0.5,0.5};

array<float,2> dt_angle = {0.1,0.1};
float past_time_angle = 0;

array<float,2> past_err = {0,0};
float past_time_err = 0;

array<float,2> accumulated_err = {0,0};

float kp = 1.0;
float ki = 0;
float kd = 0.04;

array<float,2> wall = {0.08084, 0.06543}; // thickness in x and y

float V_max = 24;

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

double f(double x, int i, double eat, double ebt);
double g(double tms, int i);
array<float,2> expected_position(double t);
void apply_voltage(double t);
void setup_coefficients();
void read_motor_angles();
array<float,2> theta_to_xy(array<float,2> theta);
array<float,2> xy_to_theta(array<float,2> xy);
void set_motor_pwms(float left, float right);
void home_table();
void setup_mode();
void send_motor_pos();
bool update_goal();

void setup() {
  Serial.begin(115200);
  SPI.beginTransaction(SPISettings(115200, MSBFIRST, SPI_MODE1));

  while (!Serial.available()) {
    continue;
  }

  Serial.println("ready to home");

  home_table();

  read_motor_angles();
  read_motor_angles();
  read_motor_angles();

  setup_mode();
}

void setup_mode() {
  String mode_str = Serial.readStringUntil('\n');
  while (mode_str.length() == 0) {
    read_motor_angles();
    mode_str = Serial.readStringUntil('\n');
  }
  mode = mode_str.toInt();
  Serial.println(mode);

  if (mode == 3 || mode == 4 || mode == 5) {
    setup_coefficients();
  }

  if (mode != 1) {
    String continue_str = Serial.readStringUntil('\n');
    while (continue_str.length() == 0) {
        read_motor_angles();
        continue_str = Serial.readStringUntil('\n');
    }
  }

  // End of choose mode in agent_processing

  String continue_str = Serial.readStringUntil('\n');
  while (continue_str.length() == 0) {
    read_motor_angles();
    send_motor_pos();
    continue_str = Serial.readStringUntil('\n');
  }

  if (mode != 1) {
    while (!update_goal()) {
      read_motor_angles();
    }
  }

  past_time_angle = micros() / 1e6;
  past_time_err = micros() / 1e6;
}

bool update_goal() {
  if (mode == 5 || mode == 4 || mode == 3) {
    String jsonString = Serial.readStringUntil('\n');
    
    if (jsonString.length() > 0) {
        // Parse JSON data
        StaticJsonDocument<512> doc;
        DeserializationError error = deserializeJson(doc, jsonString);
        
      if (!error) {
        if (doc["init"]) {
          set_motor_pwms(0,0);
          setup_mode();
          return true;
        }

        t_init = micros() / 1e6;
        // Extract data from JSON
        vt_1[0] = doc["vt_1"][0];
        vt_1[1] = doc["vt_1"][1];
        
        vt_2[0] = doc["vt_2"][0];
        vt_2[1] = doc["vt_2"][1];
        
        Vf[0] = doc["Vf"][0];
        Vf[1] = doc["Vf"][1];
        
        C2[0] = doc["C2"][0];
        C2[1] = doc["C2"][1];
        
        C3[0] = doc["C3"][0];
        C3[1] = doc["C3"][1];
        
        C4[0] = doc["C4"][0];
        C4[1] = doc["C4"][1];

        return true;
      }
    }
  }
  else if (mode == 2) {
    String jsonString = Serial.readStringUntil('\n');
    
    if (jsonString.length() > 0) {
        // Parse JSON data
        StaticJsonDocument<512> doc;
        DeserializationError error = deserializeJson(doc, jsonString);
        
      if (!error) {
        if (doc["init"]) {
          set_motor_pwms(0,0);
          setup_mode();
          return true;
        }

        t_init = micros() / 1e6;
        // Extract data from JSON
        xf[0] = doc["xf"][0];
        xf[1] = doc["xf"][1];

        return true;
      }
    }
  }
  return false;
}

void send_motor_pos() {
  float dt_sum = dt_angle[0] + dt_angle[1];
  array<float,2> xy_vel = {(xy_pos[0] - pp_xy_pos[0]) / (dt_sum), (xy_pos[1] - pp_xy_pos[1]) / (dt_sum)};
  array<float,2> xy_acc = {(dt_angle[0] * pp_xy_pos[0] - dt_sum * p_xy_pos[0] + dt_angle[1]*xy_pos[0])*2/(dt_angle[0]*dt_angle[1]*dt_sum),\
    (dt_angle[0] * pp_xy_pos[0] - dt_sum * p_xy_pos[0] + dt_angle[1]*xy_pos[0])*2/(dt_angle[0]*dt_angle[1]*dt_sum)};

  String jsonString = "{\"pos\":[";
  jsonString += String(xy_pos[0], 6) + "," + String(xy_pos[1], 6);
  jsonString += "],\"vel\":[";
  jsonString += String(xy_vel[0], 6) + "," + String(xy_vel[1], 6);
  jsonString += "],\"acc\":[";
  jsonString += String(xy_acc[0], 6) + "," + String(xy_acc[1], 6);
  jsonString += "]}";
  
  Serial.println(jsonString);
}

void loop() {
  if (mode == 1) {
    String jsonString = Serial.readStringUntil('\n');
      
    if (jsonString.length() > 0) {
      set_motor_pwms(0,0);
      setup_mode();
    }

    read_motor_angles();
  }
  else {
    update_goal();

    read_motor_angles();
    float dt = micros() / 1e6 - t_init;
    apply_voltage(dt);
  }

  send_motor_pos();
}

void apply_voltage(double t) {

  array<float,2> Vxy = {0,0};
  array<float,2> V = {0,0};

  if (mode == 5 || mode == 4) {
    if (t<vt_1[0]) {
        Vxy[0] = Vf[0];
    } else {
        Vxy[0] = -Vf[0];
    }

    if (t<vt_2[0]) {
        Vxy[1] = Vf[1];
    } else {
        Vxy[1] = -Vf[1];
    }

    V[0] = Vxy[0]/2 - Vxy[1]/2;
    V[1] = -Vxy[0]/2 - Vxy[1]/2;
  }

  if (mode != 4) {
    array<float,2> theta_hat = expected_position(t);
    array<float,2> err = {theta_hat[0] - angle[0], theta_hat[1] - angle[1]};
    float current_time = micros() / 1e6;
    float dt = current_time - past_time_err;
    past_time_err = current_time;

    V[0] += kp * err[0];
    V[1] += kp * err[1];

    V[0] += kd * (err[0] - past_err[0]) / dt;
    V[1] += kd * (err[1] - past_err[1]) / dt;

    accumulated_err[0] += err[0] * dt;
    accumulated_err[1] += err[1] * dt;

    V[0] += ki * accumulated_err[0];
    V[1] += ki * accumulated_err[1];
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


  set_motor_pwms(V[0] * 100/V_max, V[1] * 100/V_max);
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

double f(double x, int i, double eat, double ebt) {
  return CE[i][0] * eat + CE[i][1] * ebt + CE[i][2] * x + CE[i][3];
}

double g(double tms, int i) {
  if (tms < 0) {
    return 0;
  }
  double eat = exp(ab[i][0] * tms);
  double ebt = exp(ab[i][1] * tms);
  return f(tms, i, eat, ebt);
}

array<float,2> expected_position(double t) {
  if (mode == 2) {
    return xy_to_theta(xf);
  }
  array<float,2> xy_posisition;
  for (int i = 0; i < 2; i++) {
    double eat = exp(ab[i][0] * t);
    double ebt = exp(ab[i][1] * t);
    
    double A2 = A2CE[i][0] * eat + A2CE[i][1] * ebt + A2CE[i][2];
    double A3 = A3CE[i][0] * eat + A3CE[i][1] * ebt;
    double A4 = A4CE[i][0] * eat + A4CE[i][1] * ebt;
    
    double f_t = f(t, i, eat, ebt);
    double g_a1 = g(t - vt_1[i], i);
    double g_a2 = g(t - vt_2[i], i);
    
    xy_posisition[i] = Vf[i] * pullyR * (f_t - 2 * g_a1 + g_a2) + C2[i] * A2 + C3[i] * A3 + C4[i] * A4;
  }

  return xy_to_theta(xy_posisition);
}

void read_motor_angles() {
  pp_angle = p_angle;
  p_angle = angle;

  dt_angle[1] = dt_angle[0];

  array<float,2> encoder;

  u_int16_t serial_response; // incoming byte from the SPI
  int chips[2] = {ENC_CHIP_SELECT_LEFT, ENC_CHIP_SELECT_RIGHT};

  digitalWrite(chips[0], LOW);
  serial_response = SPI.transfer16(0x3FFF);
  digitalWrite(chips[0], HIGH);
  encoder[0] = (serial_response & 0b0011111111111111) * 360.0 / 16384;

  if(encoder[0] - p_angle[0] > ROLLOVER_ANGLE_DEGS) {
      revolutions[0] += 1;
  } else if(p_angle[0] - encoder[0] > ROLLOVER_ANGLE_DEGS) {
      revolutions[0] -= 1;
  }

  digitalWrite(chips[1], LOW);
  serial_response = SPI.transfer16(0x3FFF);
  digitalWrite(chips[1], HIGH);
  encoder[1] = (serial_response & 0b0011111111111111) * 360.0 / 16384;

  if(encoder[1] - p_angle[1] > ROLLOVER_ANGLE_DEGS) {
      revolutions[1] += 1;
  } else if (p_angle[1] - encoder[1] > ROLLOVER_ANGLE_DEGS) {
      revolutions[1] -= 1;
  }

  angle[0] = -encoder[0] + 360.0 * revolutions[0] + offset[0];
  angle[1] = -encoder[1] + 360.0 * revolutions[1] + offset[1];

  pp_xy_pos = p_xy_pos;
  p_xy_pos = xy_pos;
  xy_pos = theta_to_xy(angle);

  float current_time = micros() / 1e6;
  dt_angle[0] = current_time - past_time_angle;
  past_time_angle = current_time;
  
}

array<float,2> theta_to_xy(array<float,2> theta) {
  array<float, 2> xy_position;
  xy_position[0] = (theta[0] - theta[1]) * PULLEY_RADIUS * PI / 360;
  xy_position[1] = (-theta[0] - theta[1]) * PULLEY_RADIUS * PI / 360;

  return xy_position;
}

array<float,2> xy_to_theta(array<float,2> xy) {
  array<float, 2> theta;
  theta[0] = (xy[0] - xy[1]) * 360 / (2*PI*PULLEY_RADIUS);
  theta[1] = (-xy[0] - xy[1]) * 360 / (2*PI*PULLEY_RADIUS);

  return theta;
}

void set_motor_pwms(float left, float right) {
  if (left >= 0) {
      digitalWrite(LEFT_MOTOR_DIR_PIN, LOW);
  } else {
      digitalWrite(LEFT_MOTOR_DIR_PIN, HIGH);
  }
  pwm_start(LEFT_MOTOR_PWM_PIN, PWM_FREQ_HZ, floor(abs(left) / 100.0 * DUTY_CYCLE_CONVERSION), RESOLUTION_10B_COMPARE_FORMAT);

  if (right >= 0) {
      digitalWrite(RIGHT_MOTOR_DIR_PIN, LOW);
  } else {
      digitalWrite(RIGHT_MOTOR_DIR_PIN, HIGH);
  }
  pwm_start(RIGHT_MOTOR_PWM_PIN, PWM_FREQ_HZ, floor(abs(right) / 100.0 * DUTY_CYCLE_CONVERSION), RESOLUTION_10B_COMPARE_FORMAT);
}


void home_table() {
  //Home Y
  array<float,2> x_axis;
  array<float,2> y_axis;

  String on_x_axis = Serial.readStringUntil('\n');
  while (on_x_axis.length() == 0) {
    read_motor_angles();
    on_x_axis = Serial.readStringUntil('\n');
  }
  read_motor_angles();
  x_axis = angle;

  Serial.println("confirmation of X");

  String on_y_axis = Serial.readStringUntil('\n');
  while (on_y_axis.length() == 0) {
    read_motor_angles();
    on_y_axis = Serial.readStringUntil('\n');
  }
  read_motor_angles();
  y_axis = angle;

  Serial.println("confirmation of Y");

  //xy_pos[0] = (theta_l - theta_r) * PULLEY_RADIUS * PI / 360;
  //xy_pos[1] = (-theta_l - theta_r) * PULLEY_RADIUS * PI / 360;

  //mallet_r + wall[0] = ((y_axis[0] + offset[0]) - (y_axis[1]+offset[1])) * PULLEY_RADIUS * PI/360
  //mallet_r + wall[1] = -((x_axis[0] + offset[0]) + (x_axis[1]+offset[1])) * PULLY_RADIUS * PI/360

  //2*mallet_r + wall[0]+wall[1] = (y_axis[0]-x_axis[0] - y_axis[1] - 2*offset[1] - x_axis[1]) * PULLEY_RADIUS * PI/360
  offset[1] = (360 * (2*MALLET_RADIUS+wall[0]+wall[1]) / (PULLEY_RADIUS * PI) - y_axis[0]+x_axis[0]+y_axis[1]+x_axis[1])/(-2);

  //wall[0] - wall[1] = (y_axis[0] + x_axis[0] + 2*offset[0] - y_axis[1] + x_axis[1]) * PULLY_RADIUS * PI/360
  offset[0] = (360*(wall[0] - wall[1])/(PULLEY_RADIUS*PI) - y_axis[0] - x_axis[0] + y_axis[1] - x_axis[1])/2;

  // Serial.println("Fully Homed");
}
