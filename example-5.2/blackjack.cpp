#include <iostream>
#include <random>
#include <tuple>
#include <functional>
#include <array>
using namespace std;

default_random_engine rng(random_device{}());
// default_random_engine rng(42);

const int k_blackjack = 21;
const int k_ace_value = 11;
const int k_dealer_stick = 17;
const int k_n_episodes = 500000;

enum class Action {stick, hit};

template<typename T, typename Hasher=hash<T>>
void hash_combine(size_t& seed, const T& value) {
  Hasher hasher;
  seed ^= hasher(value) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

int draw_card() {
  uniform_int_distribution<int> unif(0,51);
  return unif(rng);
}

Action random_action() {
  uniform_int_distribution<int> unif(0,1);
  return unif(rng) == 0? Action::stick : Action::hit;
}

struct State {
  int player_sum;
  int usable_aces;
  int dealer_sum;
};

bool operator==(const State& s1, const State& s2) {
  return s1.player_sum == s2.player_sum and
         s1.usable_aces == s2.usable_aces and
         s1.dealer_sum == s2.dealer_sum;
}

struct StateHash {
  size_t operator()(const State& state) const {
    size_t seed = 0;
    hash_combine(seed, state.player_sum);
    hash_combine(seed, state.usable_aces);
    hash_combine(seed, state.dealer_sum);
    return seed;
  }
};

class SampleAverage {
public:
  SampleAverage() : _avg(0), _n(0) {

  }

  double get_average() const {
    return _avg;
  }

  void update(double v) {
    _n += 1;
    _avg += (v-_avg)/_n;
    // cout << v << ' ' << _n << ' ' << _avg << endl;
  }
private:
  double _avg;
  int _n;
};

template <typename Value>
using StateMap = unordered_map<State,Value,StateHash>;

void deal(int& sum, int& usable_aces) {
  int card_value = min(draw_card()%13 + 1, 10);
  if (card_value == 1) {
    card_value = k_ace_value;
    usable_aces += 1;
  }
  sum += card_value;
  while (sum > k_blackjack and usable_aces) {
    sum -= k_ace_value - 1;
    usable_aces -= 1;
  }
}

class Environment {
public:
  Environment() : _player_sum(0), _dealer_sum(0), _player_usable_aces(0), _dealer_usable_aces(0), _result(0), _done(false) {
    deal(_player_sum, _player_usable_aces);
    deal(_player_sum, _player_usable_aces);
    deal(_dealer_sum, _dealer_usable_aces);
    if (_player_sum == k_blackjack) rollout_dealer();
    else if (_player_sum > k_blackjack) {
      _result = -1;
      _done = true;
    }
  }

  void step(Action action) {
    if (action == Action::hit) {
      deal(_player_sum, _player_usable_aces);
      if (_player_sum == k_blackjack) rollout_dealer();
      else if (_player_sum > k_blackjack) {
        _result = -1;
        _done = true;
      }
    }
    else rollout_dealer();
  }

  State get_state() const {
    return {_player_sum, _player_usable_aces, _dealer_sum};
  }

  double get_result() const {
    return _result;
  }

  bool is_done() const {
    return _done;
  }

private:

  void rollout_dealer() {
    while (_dealer_sum < k_dealer_stick)
      deal(_dealer_sum, _dealer_usable_aces);
    if (_dealer_sum > k_blackjack or _player_sum > _dealer_sum) _result = 1;
    else if (_dealer_sum > _player_sum) _result = -1;
    _done = true;
  }

  int _player_sum;
  int _dealer_sum;
  int _player_usable_aces;
  int _dealer_usable_aces;
  double _result;
  bool _done;
};

ostream& operator<<(ostream& out, const State& state) {
  return out << '(' << state.player_sum << ',' << state.usable_aces << ',' << state.dealer_sum << ')';
}

class MCAgent {
public:
  struct Q {
    SampleAverage hit, stick;
  };

  void set_exploring_start() {
    _es = true;
  }

  Action query_next_action(const State& state) {
    Action ret;
    if (state.player_sum < 12)
      ret = Action::hit;
    else if (_es) {
      _es = false;
      ret = random_action();
    }
    else {
      auto it = _q.find(state);
      if (it != _q.end()) {
        const Q& q_s = it->second;
        ret = q_s.hit.get_average() > q_s.stick.get_average()? Action::hit : Action::stick;
      }
      else
        ret = random_action();
    }
    return ret;
  }

  void backup(const vector<tuple<State,Action>>& trajectory, double result) {
    for (const auto&[state, action] : trajectory) {
      if (action == Action::hit)
        _q[state].hit.update(result);
      else
        _q[state].stick.update(result);
    }
  }

  const StateMap<Q>& get_q() const {
    return _q;
  }

private:
  StateMap<Q> _q;
  bool _es;
};

int main() {
  // while (true) {
  //   Environment env;
  //   while (not env.is_done()) {
  //     cout << env.get_state() << " [h/s] ";
  //     char cmd;
  //     cin >> cmd;
  //     Action action = cmd == 'h'? Action::hit : Action::stick;
  //     env.step(action);
  //   }
  //   cout << env.get_state() << ", result: " << env.get_result() << endl << "----" << endl;
  // }
  MCAgent agent;
  for (int i = 0; i < k_n_episodes; ++i) {
    Environment env;
    vector<tuple<State,Action>> trajectory;
    agent.set_exploring_start();
    while (not env.is_done()) {
      State state = env.get_state();
      Action act = agent.query_next_action(state);
      env.step(act);
      trajectory.emplace_back(state, act);
    }
    agent.backup(trajectory, env.get_result());
    // if (i%1000 == 0) cerr << "Iter. " << i << ": " << avg_return.get_average() << '\n';
    // cout << i << ',' << avg_return.get_average();
  }
  cout << "X,Y,Z" << endl;
  cerr << "X,Y,Z" << endl;
  for (const auto&[state,q] : agent.get_q()) {
    if (state.player_sum > 11) {
      ostream& out = state.usable_aces? cout : cerr;
      out << state.player_sum << ',' <<
             state.dealer_sum << ',' <<
             max(q.hit.get_average(), q.stick.get_average()) <<
             endl;
    }
  }
}
