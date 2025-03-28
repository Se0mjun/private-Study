# 정책기반 강화학습 매트랩 코드 예시

% ------------------------------
% 정책 기반 강화학습 (REINFORCE) + 보상 shaping + 시각화
% ------------------------------

clear; clc;

% 환경 설정
rows = 4; cols = 4;
numStates = rows * cols;
numActions = 4; % 상(1), 하(2), 좌(3), 우(4)
goalState = sub2ind([rows, cols], 4, 4); % 목표 위치 (4,4)

% 정책 파라미터 초기화
theta = randn(numStates, numActions);
gamma = 0.99;     % 할인율
alpha = 0.01;     % 학습률
episodes = 1000;  % 학습 횟수

% 안정적인 softmax 함수
softmax = @(x) exp(x - max(x)) / sum(exp(x - max(x)));

% 목표 좌표
[goalRow, goalCol] = ind2sub([rows, cols], goalState);

% 학습 루프
for ep = 1:episodes
    state = randi(numStates); % 랜덤 시작
    trajectory = [];

    while state ~= goalState
        probs = softmax(theta(state, :));
        action = randsample(1:numActions, 1, true, probs);
        nextState = move(state, action, rows, cols);

        % 보상 shaping
        [curRow, curCol] = ind2sub([rows, cols], state);
        [nextRow, nextCol] = ind2sub([rows, cols], nextState);
        distBefore = abs(goalRow - curRow) + abs(goalCol - curCol);
        distAfter  = abs(goalRow - nextRow) + abs(goalCol - nextCol);

        reward = -1;
        if nextState == goalState
            reward = 100;
        elseif distAfter < distBefore
            reward = 1;
        elseif distAfter > distBefore
            reward = -2;
        end

        % trajectory 저장
        trajectory(end+1).state = state;
        trajectory(end).action = action;
        trajectory(end).reward = reward;

        state = nextState;
    end

    % 리턴 계산 및 정책 업데이트
    G = 0;
    for t = length(trajectory):-1:1
        G = gamma * G + trajectory(t).reward;
        s = trajectory(t).state;
        a = trajectory(t).action;
        probs = softmax(theta(s, :));
        grad = -probs;
        grad(a) = grad(a) + 1;
        theta(s, :) = theta(s, :) + alpha * G * grad;
    end
end

% ------------------------------
% 학습 후 경로 시각화
% ------------------------------

figure;
hold on;
axis([0 cols 0 rows]);
xticks(0:cols); yticks(0:rows);
grid on;
title('정책 기반 강화학습 + 보상 shaping 결과');
xlabel('열'); ylabel('행');

% 목표 위치 표시
rectangle('Position', [cols-1, 0, 1, 1], 'FaceColor', [0.2 1 0.2]);
text(cols-0.5, 0.5, 'Goal', 'HorizontalAlignment', 'center');

% 경로 시뮬레이션 (시작: (1,1))
startState = sub2ind([rows, cols], 1, 1);
state = startState;
pathX = [];
pathY = [];

for step = 1:50
    [r, c] = ind2sub([rows, cols], state);
    pathX(end+1) = c - 0.5;
    pathY(end+1) = rows - r + 0.5;

    probs = softmax(theta(state, :));
    [~, action] = max(probs);
    nextState = move(state, action, rows, cols);

    if state == nextState
        break;
    end

    state = nextState;

    if state == goalState
        [r, c] = ind2sub([rows, cols], state);
        pathX(end+1) = c - 0.5;
        pathY(end+1) = rows - r + 0.5;
        break;
    end
end

% 경로 출력
plot(pathX, pathY, 'bo-', 'LineWidth', 2);
legend('경로');

if state == goalState
    disp("🎯 목표 도달 성공!");
else
    disp("❌ 아직 도달 못 함.");
end

% ------------------------------
% 상태 이동 함수 (move)
% ------------------------------
function next = move(state, action, rows, cols)
    [r, c] = ind2sub([rows, cols], state);
    switch action
        case 1, r = max(r-1, 1);      % ↑ 상
        case 2, r = min(r+1, rows);   % ↓ 하
        case 3, c = max(c-1, 1);      % ← 좌
        case 4, c = min(c+1, cols);   % → 우
    end
    next = sub2ind([rows, cols], r, c);
end
