#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
using namespace std;

constexpr int s_max = 100;
constexpr double p_h = 0.4;
constexpr double eps = 0.0005;
constexpr double inf = -numeric_limits<double>::infinity();


void relax(vector<double>& v, int s) {
  for (int a = 1; a <= min(s, s_max-s); ++a) {
    double q_s_a = p_h*v[s+a] + (1-p_h)*v[s-a];
    v[s] = max(v[s], q_s_a);
  }
}

double sweep_once(vector<double>& v) {
  double error = 0;
  for (int s = 0; s <= s_max; ++s) {
    double v_s_prev = v[s];
    relax(v, s);
    error = max(error, fabs(v_s_prev - v[s]));
  }
  return error;
}

vector<int> best_actions(vector<double>& v, int s) {
  vector<int> best;
  for (int a = 1; a <= min(s, s_max-s); ++a) {
    double q_s_a = p_h*v[s+a] + (1-p_h)*v[s-a];
    if (fabs(v[s]-q_s_a) < 2*eps) {
      best.push_back(a);
    }
  }
  return best;
}

int main() {
  vector<double> v(s_max+1);
  v[s_max] = 1;
  double error = sweep_once(v);
  cerr << "Error: " << error << endl;
  while (error > eps) {
    error = sweep_once(v);
    cerr << "Error: " << error << endl;
  }
  for (int s = 0; s <= s_max; ++s) {
    cout << s << ',' << v[s] << ",Best actions:";
    for (int a : best_actions(v, s)) cout << ' ' << a;
    cout << endl;
  }
}


