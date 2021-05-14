# Random Mario Map playing RL
건국대학교 오픈소스소프트웨어프로젝트1 수업의 텀프로젝트 레포입니다. 강화학습을 기반으로 한 랜덤성있는 마리오게임을 플레이하는 것이 목적입니다.


## Project Structure
1. RL (Actor-Critic based)
2. Env (Gym Env)
3. Mario Game - 원본 pygame 은 Meth-Meth-Method's [super mario game](https://github.com/meth-meth-method/super-mario/) 이고,
이를 바탕으로 Random map 생성.

## How to Run

* ```shell
  $ pip install -r requirements.txt
  $ python RL/Agent/a3c_baseline_thread.py
  ```

## Dependencies
