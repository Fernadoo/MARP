#!/usr/bin/env python3
# Modified based on USC's MAPF class project
from matplotlib.patches import Circle, Rectangle
# from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import animation
from copy import deepcopy


# Colors = ['green', 'blue', 'orange']
Colors = list(mcolors.CSS4_COLORS)


class Animation:
    def __init__(self, agents, my_map, starts, goals, history, FPS):
        self.my_map = np.flip(np.transpose(my_map), 1)
        self.starts = []
        for start in starts:
            self.starts.append((start[1], len(self.my_map[0]) - 1 - start[0]))
        self.goals = []
        for goal in goals:
            self.goals.append((goal[1], len(self.my_map[0]) - 1 - goal[0]))
        self.paths = []
        if history:
            for i in range(len(starts)):
                self.paths.append([])
                for state in history:
                    self.paths[-1].append(
                        (state[i][1],
                         len(self.my_map[0]) - 1 - state[i][0])
                    )

        aspect = len(self.my_map) / len(self.my_map[0])

        self.fig = plt.figure(frameon=False, figsize=(4 * aspect, 4))
        self.ax = self.fig.add_subplot(111, aspect='equal')
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1,
                                 wspace=None, hspace=None)
        # self.ax.set_frame_on(False)

        self.patches = []
        self.artists = []
        self.agents = dict()
        self.agent_names = dict()
        # create boundary patch

        x_min = -0.5
        y_min = -0.5
        x_max = len(self.my_map) - 0.5
        y_max = len(self.my_map[0]) - 0.5
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        self.patches.append(
            Rectangle((x_min, y_min),
                      x_max - x_min, y_max - y_min,
                      facecolor='none', edgecolor='gray')
        )
        for i in range(len(self.my_map)):
            for j in range(len(self.my_map[0])):
                if self.my_map[i][j]:
                    self.patches.append(
                        Rectangle((i - 0.5, j - 0.5),
                                  1, 1,
                                  facecolor='gray', edgecolor='gray')
                    )
                else:
                    self.patches.append(
                        Rectangle((i - 0.5, j - 0.5),
                                  1, 1,
                                  facecolor='white', edgecolor='grey')
                    )

        # create agents:
        self.T = 0
        # draw goals first
        for i, goal in enumerate(self.goals):
            self.patches.append(
                Rectangle((goal[0] - 0.25, goal[1] - 0.25),
                          0.5, 0.5,
                          facecolor=Colors[i % len(Colors)],
                          edgecolor='black',
                          alpha=0.5)
            )

        # self.sensors = dict()
        for i in range(len(self.paths)):
            name = f'{i + 1}'
            self.agents[i] = Circle((self.starts[i][0], self.starts[i][1]),
                                    0.3,
                                    facecolor=Colors[i % len(Colors)],
                                    edgecolor='black')

            self.agents[i].original_face_color = Colors[i % len(Colors)]
            self.patches.append(self.agents[i])
            self.T = max(self.T, len(self.paths[i]) - 1)
            self.agent_names[i] = self.ax.text(self.starts[i][0],
                                               self.starts[i][1],
                                               name)

            self.agent_names[i].set_horizontalalignment('center')
            self.agent_names[i].set_verticalalignment('center')
            self.artists.append(self.agent_names[i])

            # for j in range(i + 1, len(self.paths)):
            #     line = Line2D((starts[i][0], starts[j][0]),
            #                   (starts[i][1], starts[j][1]),
            #                   color='black',
            #                   linestyle='dotted',
            #                   alpha=0)
            #     self.sensors[f'{i}-{j}'] = line
            #     self.artists.append(line)
        self.FPS = FPS
        self.animation = animation.FuncAnimation(self.fig, self.animate_func,
                                                 init_func=self.init_func,
                                                 frames=int(self.T + 1) * FPS,
                                                 interval=1000 / FPS,
                                                 blit=True)

    def save(self, file_name, speed):
        self.animation.save(
            file_name,
            fps=self.FPS * speed,
            dpi=200,
            savefig_kwargs={"pad_inches": 0})  # "bbox_inches": "tight"

    @staticmethod
    def show():
        plt.show()

    def init_func(self):
        for p in self.patches:
            self.ax.add_patch(p)
        for a in self.artists:
            self.ax.add_artist(a)
        return self.patches + self.artists

    def animate_func(self, t):
        for k in range(len(self.paths)):
            pos = self.get_state(t / self.FPS, self.paths[k])
            self.agents[k].center = (pos[0], pos[1])
            self.agent_names[k].set_position((pos[0], pos[1]))

        # reset all colors
        for _, agent in self.agents.items():
            agent.set_facecolor(agent.original_face_color)

        # check collisions
        agents_array = [agent for _, agent in self.agents.items()]

        for i in range(0, len(agents_array)):
            for j in range(i + 1, len(agents_array)):
                d1 = agents_array[i]
                d2 = agents_array[j]
                pos1 = np.array(d1.center)
                pos2 = np.array(d2.center)
                if np.linalg.norm(pos1 - pos2) < 0.7:
                    d1.set_facecolor('red')
                    d2.set_facecolor('red')

        return self.patches + self.artists

    @staticmethod
    def get_state(t, path):
        if int(t) <= 0:
            return np.array(path[0])
        elif int(t) >= len(path):
            return np.array(path[-1])
        else:
            pos_last = np.array(path[int(t) - 1])
            pos_next = np.array(path[int(t)])
            pos = (pos_next - pos_last) * (t - int(t)) + pos_last
            return pos


class StreamAnimation(Animation):
    """docstring for StreamAnimation"""

    def __init__(self, agents, my_map, starts, goal_streams, history, FPS):
        self.my_map = np.flip(np.transpose(my_map), 1)
        self.starts = []
        for start in starts:
            self.starts.append((start[1], len(self.my_map[0]) - 1 - start[0]))

        self.goal_streams = []
        for i, goals in enumerate(goal_streams):
            self.goal_streams.append([])
            for g in goals:
                self.goal_streams[-1].append((g[1], len(self.my_map[0]) - 1 - g[0]))
        self.goal_streams_backup = deepcopy(self.goal_streams)

        self.paths = []
        if history:
            for i in range(len(starts)):
                self.paths.append([])
                for state in history:
                    self.paths[-1].append((state[i][1], len(self.my_map[0]) - 1 - state[i][0]))

        aspect = len(self.my_map) / len(self.my_map[0])

        self.fig = plt.figure(frameon=False, figsize=(4 * aspect, 4))
        self.ax = self.fig.add_subplot(111, aspect='equal')
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1,
                                 wspace=None, hspace=None)
        # self.ax.set_frame_on(False)

        self.patches = []
        self.artists = []
        self.agents = dict()
        self.agent_names = dict()
        # create boundary patch

        x_min = -0.5
        y_min = -0.5
        x_max = len(self.my_map) - 0.5
        y_max = len(self.my_map[0]) - 0.5
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        self.patches.append(
            Rectangle((x_min, y_min),
                      x_max - x_min, y_max - y_min,
                      facecolor='none', edgecolor='gray')
        )
        for i in range(len(self.my_map)):
            for j in range(len(self.my_map[0])):
                if self.my_map[i][j]:
                    self.patches.append(
                        Rectangle((i - 0.5, j - 0.5),
                                  1, 1,
                                  facecolor='gray', edgecolor='gray')
                    )
                else:
                    self.patches.append(
                        Rectangle((i - 0.5, j - 0.5),
                                  1, 1,
                                  facecolor='white', edgecolor='grey')
                    )

        # create agents:
        self.T = 0
        # draw goals first
        self.next_goals = {}
        for i, goals in enumerate(self.goal_streams):
            goal = goals[0]
            self.next_goals[i] = Rectangle((goal[0] - 0.25, goal[1] - 0.25),
                                           0.5, 0.5,
                                           facecolor=Colors[i % len(Colors)],
                                           edgecolor='black',
                                           alpha=0.5)
            self.patches.append(self.next_goals[i])

        for i in range(len(self.paths)):
            name = f'{i + 1}'
            self.agents[i] = Circle((self.starts[i][0], self.starts[i][1]),
                                    0.3,
                                    facecolor=Colors[i % len(Colors)],
                                    edgecolor='black')

            self.agents[i].original_face_color = Colors[i % len(Colors)]
            self.patches.append(self.agents[i])
            self.T = max(self.T, len(self.paths[i]) - 1)
            self.agent_names[i] = self.ax.text(self.starts[i][0],
                                               self.starts[i][1],
                                               name)

            self.agent_names[i].set_horizontalalignment('center')
            self.agent_names[i].set_verticalalignment('center')
            self.artists.append(self.agent_names[i])

        self.FPS = FPS
        self.animation = animation.FuncAnimation(self.fig, self.animate_func,
                                                 init_func=self.init_func,
                                                 frames=int(self.T + 1) * FPS,
                                                 interval=1000 / FPS,
                                                 blit=True)

    def init_func(self):
        self.goal_streams = deepcopy(self.goal_streams_backup)
        for k in self.agents:
            new_goal = self.goal_streams[k][0]
            self.next_goals[k].set_xy((new_goal[0] - 0.25, new_goal[1] - 0.25))
        for p in self.patches:
            self.ax.add_patch(p)
        for a in self.artists:
            self.ax.add_artist(a)
        return self.patches + self.artists

    def animate_func(self, t):
        for k in range(len(self.paths)):
            pos = self.get_state(t / self.FPS, self.paths[k])
            self.agents[k].center = (pos[0], pos[1])
            self.agent_names[k].set_position((pos[0], pos[1]))

            if np.allclose(pos, self.goal_streams[k][0]) and len(self.goal_streams[k]) > 1:
                self.goal_streams[k].pop(0)
                new_goal = self.goal_streams[k][0]
                self.next_goals[k].set_xy((new_goal[0] - 0.25, new_goal[1] - 0.25))

        # reset all colors
        for _, agent in self.agents.items():
            agent.set_facecolor(agent.original_face_color)

        # check collisions
        agents_array = [agent for _, agent in self.agents.items()]

        for i in range(0, len(agents_array)):
            for j in range(i + 1, len(agents_array)):
                d1 = agents_array[i]
                d2 = agents_array[j]
                pos1 = np.array(d1.center)
                pos2 = np.array(d2.center)
                if np.linalg.norm(pos1 - pos2) < 0.7:
                    d1.set_facecolor('red')
                    d2.set_facecolor('red')

        return self.patches + self.artists


class WarehouseAnimation(StreamAnimation):
    """docstring for WarehouseAnimcation"""

    def __init__(self, agents, my_map, starts, goal_streams, history, aux, FPS):
        super().__init__(agents, my_map, starts, goal_streams, history, FPS)

        # batteries
        self.batteries = []
        for i in range(len(agents)):
            self.batteries.append([])
            for step in aux['batteries']:
                self.batteries[-1].append(step[i])

        # set agent battery text
        self.battery_text = {}
        for i in range(len(agents)):
            self.battery_text[i] = self.ax.text(self.starts[i][0] + 0.35,
                                                self.starts[i][1] + 0.35,
                                                self.batteries[i][0],
                                                horizontalalignment='center',
                                                verticalalignment='center',
                                                fontsize='x-small')
            self.artists.append(self.battery_text[i])

    def animate_func(self, t):
        out_of_bat = {agent: False for agent in self.agents}
        for k in range(len(self.paths)):
            pos = self.get_state(t / self.FPS, self.paths[k])
            self.agents[k].center = (pos[0], pos[1])
            self.agent_names[k].set_position((pos[0], pos[1]))
            self.battery_text[k].set_position((pos[0] + 0.35, pos[1] + 0.35))

            if np.allclose(pos, self.goal_streams[k][0]) and len(self.goal_streams[k]) > 1:
                self.goal_streams[k].pop(0)
                new_goal = self.goal_streams[k][0]
                self.next_goals[k].set_xy((new_goal[0] - 0.25, new_goal[1] - 0.25))

            t2i = max(int(t / self.FPS) - 1, 0)
            if np.allclose(pos, self.paths[k][t2i]):
                self.battery_text[k].set(text=self.batteries[k][t2i],
                                         horizontalalignment='center',
                                         verticalalignment='center')
                if self.batteries[k][t2i] == 0:
                    out_of_bat[k] = True

        # reset all colors
        for _, agent in self.agents.items():
            agent.set_facecolor(agent.original_face_color)

        # check collisions
        agents_array = [agent for _, agent in self.agents.items()]

        for i in range(0, len(agents_array)):
            for j in range(i + 1, len(agents_array)):
                d1 = agents_array[i]
                d2 = agents_array[j]
                pos1 = np.array(d1.center)
                pos2 = np.array(d2.center)
                if np.linalg.norm(pos1 - pos2) < 0.7:
                    d1.set_facecolor('red')
                    d2.set_facecolor('red')

        for i in self.agents:
            if out_of_bat[i]:
                self.agents[i].set_facecolor('lightcoral')

        return self.patches + self.artists
