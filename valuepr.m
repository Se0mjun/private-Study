# 가치기반 강화학습 매트랩 예시 코드

% Grid World 환경 정의
rows = 5;
cols = 5;
numStates = rows * cols;
numActions = 4;  % 상(1), 하(2), 좌(3), 우(4)

% 파라미터
alpha = 0.1;    % 학습률
gamma = 0.9;    % 할인율
epsilon = 0.1;  % 탐험 확률
episodes = 1000;

% 보상 및 상태 정의
goalState = sub2ind([rows, cols], 5, 5);
Q = zeros(numStates, numActions);
reward = -1 * ones(rows, cols);
reward(5,5) = 100;  % 목표지점 보상

% 상태 전이 함수
getNextState = @(s, a) max(1, min(numStates, ...
    move(s, a, rows, cols)));

% Q-learning 알고리즘
for ep = 1:episodes
    state = randi(numStates);  % 임의 시작 상태
    while state ~= goalState
        if rand < epsilon
            action = randi(numActions);
        else
            [~, action] = max(Q(state, :));
        end

        nextState = getNextState(state, action);
        [r, c] = ind2sub([rows, cols], nextState);
        rwd = reward(r, c);

        Q(state, action) = Q(state, action) + ...
            alpha * (rwd + gamma * max(Q(nextState, :)) - Q(state, action));

        state = nextState;
    end
end

% 최적 정책 출력
disp("Optimal Policy:");
for s = 1:numStates
    [~, bestAction] = max(Q(s,:));
    fprintf('State %2d: Best Action = %d\n', s, bestAction);
end

% 상태 이동 함수
function next = move(state, action, rows, cols)
    [r, c] = ind2sub([rows, cols], state);
    switch action
        case 1  % 상
            r = max(r-1, 1);
        case 2  % 하
            r = min(r+1, rows);
        case 3  % 좌
            c = max(c-1, 1);
        case 4  % 우
            c = min(c+1, cols);
    end
    next = sub2ind([rows, cols], r, c);
end

% -------------------------------
% 시각화: 최적 경로 보여주기
% -------------------------------
figure;
hold on;
axis([0 cols 0 rows]);
xticks(0:cols); yticks(0:rows);
grid on;
title('최적 경로 시각화');
xlabel('열'); ylabel('행');

% 목표 지점 표시
rectangle('Position', [cols-1, 0, 1, 1], 'FaceColor', [0.2 1 0.2]);
text(cols-0.5, 0.5, 'Goal', 'HorizontalAlignment', 'center');

% 에이전트 초기 위치
startState = sub2ind([rows, cols], 1, 1);  % 시작점 (1,1)
state = startState;

pathX = [];
pathY = [];

while state ~= goalState
    [r, c] = ind2sub([rows, cols], state);
    pathX(end+1) = c - 0.5;
    pathY(end+1) = rows - r + 0.5;

    [~, action] = max(Q(state, :));
    nextState = getNextState(state, action);

    % 화살표로 방향 표시
    dr = [0, 1, 0, -1]; % 상하좌우 y 변화
    dc = [1, 0, -1, 0]; % 상하좌우 x 변화

    dx = dc(action);
    dy = -dr(action);

    quiver(c - 0.5, rows - r + 0.5, dx*0.5, dy*0.5, 0, 'r', 'LineWidth', 2);

    if state == nextState
        break; % 움직이지 못하는 경우 무한 루프 방지
    end
    state = nextState;
end

% 마지막 위치 추가
[r, c] = ind2sub([rows, cols], state);
pathX(end+1) = c - 0.5;
pathY(end+1) = rows - r + 0.5;

% 경로 연결선
plot(pathX, pathY, 'bo-', 'LineWidth', 2);
legend('경로', '위치 방향', '위치');

