import pygame as pg
import pymunk as pm
import math
import mss
import random
import numpy as np
import copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pg.init()

xScreen = 1920
yScreen = 1080
screen = pg.display.set_mode((xScreen, yScreen), pg.FULLSCREEN)
clock = pg.time.Clock()
running = True

numberOfInputs = 2 # Used in testing when trying out different NN input setups


class Limb():
    parentIndex = -1 # Index of limb to which joint is attached
    parentAnchor = -1 # Side of parent limb to which joint is attached (-1 is left)

    length = 100
    width = 10
    angleOffset = 0

    min_angle = 0
    max_angle = 0
    stiffness = 0

# Used for the fitness history graph
class Plotter():
    def __init__(self):
        self.fig_size = (3, 2)
        self.dpi = 100
        self.fig, self.ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        self.canvas = self.fig.canvas
        self.surface = None

    def PlotFitnessExport(self, fitness_history, figsize=(4, 3), dpi=100):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.clear()
        ax.plot(fitness_history, color="green")
        ax.set_title("Distance over generations")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness (Distance)")
        plt.tight_layout()
        return ax

    def PlotFitness(self, fitness_history, figsize=(4, 3), dpi=100):
        self.ax.clear()
        self.ax.plot(fitness_history, color="green")
        self.ax.set_title("Fitness over generations")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Fitness")
        plt.tight_layout()
        self.canvas.draw()  # renders into the Agg buffer

        return self.ax

    def FitnessHistory(self, fitness_history, figsize=(4, 3), dpi=100):
        self.PlotFitness(fitness_history, figsize=figsize, dpi=dpi)
        renderer = self.canvas.get_renderer()
        raw_data = bytes(renderer.buffer_rgba())
        size = self.canvas.get_width_height()

        self.surface = pg.image.frombuffer(raw_data, size, "RGBA")
        return self.surface

def DrawGame(screen, xCamera):
    # For some reason the text support is broken in this version of PyGame, have to do without live text in the simulation
    # Wanted to use this for live text showing the current generation and fitness

    # font = pg.font.SysFont('Arial', 20)
    # text = font.render('10', True, 'green', 'blue')
    # textRect = text.get_rect()
    # textRect.center = (xScreen // 2, yScreen // 2)
    #
    # screen.blit(text, textRect)

    for i in range(20):
        pg.draw.line(screen, (0, 0, 0), (i * 500 - xCamera + xScreen / 2, 0), (i * 500 - xCamera + xScreen / 2, yScreen), 10)

def PMtoPG(pos):
    return int(pos.x), yScreen - int(pos.y)

def MakeTorso(space, position=(0, 200), sizeModifier = 1):
    width, height = 20 * sizeModifier, 100 * sizeModifier
    mass = 1
    moment = pm.moment_for_box(mass, (width, height))

    body = pm.Body(mass=mass, moment=moment)
    body.position = position

    shape = pm.Poly.create_box(body, (width, height))
    shape.friction = 0.5
    shape.filter = pm.ShapeFilter(group=1)  # won't collide with limbs

    space.add(body, shape)
    return body, height

def BuildBody(space, genome, torso_body, sizeModifier = 1):
    bodies = []
    motors = []
    parentBodies = []

    for i, gene in enumerate(genome["Body"]):
        if gene.parentIndex == -1:
            parent_body = torso_body
            parent_len = 100
        else:
            parent_body = bodies[gene.parentIndex]
            parent_len = genome["Body"][gene.parentIndex].length

        anchor_on_parent = pm.Vec2d(0, gene.parentAnchor * parent_len * sizeModifier / 2)
        anchor_on_child  = pm.Vec2d(0, -gene.length * sizeModifier / 2)  # child's base

        world_anchor = parent_body.local_to_world(anchor_on_parent)

        limb_body = pm.Body(
            mass=1,
            moment=pm.moment_for_segment(
                1,
                (0, -gene.length / 2 * sizeModifier),
                (0,  gene.length / 2 * sizeModifier),
                gene.width * sizeModifier,
            )
        )
        limb_body.angle = parent_body.angle + gene.angleOffset

        rotated_child_anchor = anchor_on_child.rotated(limb_body.angle)
        limb_body.position = world_anchor - rotated_child_anchor  # ← key line

        limb_shape = pm.Segment(
            limb_body,
            (0, -gene.length / 2 * sizeModifier),
            (0,  gene.length / 2 * sizeModifier),
            gene.width * sizeModifier,
        )
        limb_shape.friction = 1
        limb_shape.filter = pm.ShapeFilter(group=1)

        joint = pm.PivotJoint(
            parent_body, limb_body,
            anchor_on_parent,
            anchor_on_child
        )
        joint.collide_bodies = False

        motor = pm.SimpleMotor(parent_body, limb_body, rate=0)
        motor.max_force = 5e5


        limit = pm.RotaryLimitJoint(
            parent_body, limb_body,
            min=-np.pi,
            max=np.pi
        )
        space.add(limit)

        space.add(limb_body, limb_shape, joint, motor)
        bodies.append(limb_body)
        motors.append(motor)
        parentBodies.append(parent_body)

    return bodies, motors, parentBodies

def DrawSpace(screen, space, xCamera):
    def to_screen(p):
        return int(p.x - xCamera + xScreen / 2), yScreen - int(p.y)

    for shape in space.shapes:
        if isinstance(shape, pm.Segment):
            body = shape.body

            a = body.local_to_world(shape.a)
            b = body.local_to_world(shape.b)

            pa = to_screen(a)
            pb = to_screen(b)

            # Draw thick line for the limb body
            color = (0, 0, 0, 50) if body.body_type == pm.Body.DYNAMIC else (0, 200, 100, 50) # Floor
            pg.draw.line(screen, color, pa, pb, max(2, int(shape.radius * 2)))

            # Draw circles at endpoints to make a pill shape
            r = max(1, int(shape.radius))
            color = (200, 0, 0, 50)
            pg.draw.circle(screen, color, pa, r)
            color = (0, 0, 0, 50)
            pg.draw.circle(screen, color, pb, r)

        elif isinstance(shape, pm.Circle):
            body = shape.body
            pos = body.local_to_world(shape.offset)
            p = to_screen(pos)
            pg.draw.circle(screen, (200, 0, 0), p, int(shape.radius))

        # Torso only
        elif isinstance(shape, pm.Poly):
            body = shape.body
            verts = [to_screen(body.local_to_world(v)) for v in shape.get_vertices()]
            pg.draw.polygon(screen, (0, 0, 0, 50), verts)

def CreateInititalSample(Nsamples):
    genomes = []
    for n in range(Nsamples):
        torso = Limb()
        torso.length = random.randint(100, 200)
        leg = Limb()
        leg.length = random.randint(100, 200)
        leg.angleOffset = random.randint(-90, 90) * math.pi / 180
        leg.parent = -1
        leg.parentAnchor = 1 if random.random() < 0.5 else -1
        body = [torso, leg]
        brain = Brain(n_limbs=2).weights

        genome = {"Body":body, "Brain":brain}
        genomes.append(genome)

    return genomes

class Brain():
    def __init__(self, n_limbs, hidden=16):
        self.n_inputs = n_limbs * numberOfInputs  # angles (sin + cos)
        self.n_hidden = 8 # Changed in sensitivity analysis
        self.n_outputs = n_limbs

        # Weight shapes
        self.w1_shape = (self.n_inputs, self.n_hidden)
        self.w2_shape = (self.n_hidden, self.n_outputs)
        self.b1_shape = (self.n_hidden,)
        self.b2_shape = (self.n_outputs,)

        # Total genome size for this body
        self.n_weights = int(
                np.prod(self.w1_shape) +
                np.prod(self.w2_shape)
        )

        self.weights = np.random.rand(self.n_weights)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Unpack flat genome into weight matrices
        idx = 0
        W1 = self.weights[idx: idx + np.prod(self.w1_shape)].reshape(self.w1_shape);
        idx += np.prod(self.w1_shape)
        W2 = self.weights[idx: idx + np.prod(self.w2_shape)].reshape(self.w2_shape);

        # biases not used anymore, inputs are never really zero and only adds unnecessary additional training parameters
        # idx += np.prod(self.w2_shape)
        # b1 = self.weights[idx: idx + self.n_hidden];
        # idx += self.n_hidden
        # b2 = self.weights[idx: idx + self.n_outputs]

        h = np.tanh(inputs @ W1)# + b1)
        out = np.tanh(h @ W2)# + b2)
        return out  # values in [-1, 1]

    def mutate(self):
        for i, weight in enumerate(self.weights):
            self.weights[i] *= random.uniform(0.8, 1.2)

def GetInputs(torso_body, limb_bodies, parentBodies):
    inputs = []
    for i, body in enumerate(limb_bodies):
        rel_angle = body.angle - parentBodies[i].angle
        rel_angle_sin = np.sin(rel_angle)
        rel_angle_cos = np.cos(rel_angle)
        # rel_vel   = body.angular_velocity # Not used anymore to prevent rapid vibration to be misused as a moving strategy
        inputs.extend([rel_angle_sin, rel_angle_cos])
    return np.array(inputs, dtype=np.float32)


def ApplyOutputs(outputs, motors, max_rate=10.0):
    smoothing = 0.8
    for motor, rate in zip(motors, outputs):
        motor.rate = smoothing * motor.rate + (1 - smoothing) * float(rate) * max_rate

def EndGeneration(creatures, genomes):
    ratio = 4

    fitnesses = []
    for i, creature in enumerate(creatures):
        fitness = creature[2].position.x
        fitness *= len(creature[0])
        fitnesses.append(fitness)

    sortedCreatures = sorted(creatures, key=lambda i: fitnesses[creatures.index(i)])
    sortedGenomes = sorted(genomes, key=lambda i: fitnesses[genomes.index(i)])

    bestCreatures = sortedCreatures[-int(len(sortedCreatures) / ratio):]
    bestGenomes = sortedGenomes[-int(len(sortedGenomes) / ratio):]

    fitnessHistory.append(bestCreatures[-1][2].position.x)
    genomes = NewGeneration(bestGenomes, ratio)
    return genomes

def NewGeneration(bestGenomes, ratio):
    genomes = []

    for g in bestGenomes:
        for i in range(ratio):
            genome = copy.deepcopy(g)
            genome = MutateGenome(genome)
            genomes.append(genome)

    return genomes

def AddLimbToBrain(genome):
    originalNumberOfLimbs = len(genome["Body"])
    for n in range(Brain(n_limbs=originalNumberOfLimbs).n_hidden):
        # Add 3 new inputs (rotation (sin + cos), velocity) at the end of W1
        genome["Brain"] = np.insert(genome["Brain"], int(originalNumberOfLimbs * 2), random.random())
        genome["Brain"] = np.insert(genome["Brain"], int(originalNumberOfLimbs * 2), random.random())
        if numberOfInputs == 3:
            genome["Brain"] = np.insert(genome["Brain"], int(originalNumberOfLimbs * 2), random.random())
        # Add 1 new output (motor power) at the end of W2
        genome["Brain"] = np.append(genome["Brain"], random.random())

    return genome

def RemoveOuterLimb(genome):
    originalNumberOfLimbs = len(genome["Body"])

    # Find all limbs that are not parents to any other limb
    parentIndices = {limb.parentIndex for limb in genome["Body"]}
    leaf_indices = [i for i in range(originalNumberOfLimbs) if i not in parentIndices]

    if not leaf_indices or originalNumberOfLimbs == 1:
        return genome

    remove_idx = random.choice(leaf_indices)

    n_inputs = originalNumberOfLimbs * numberOfInputs
    n_hidden = Brain(n_limbs=originalNumberOfLimbs).n_hidden
    n_outputs = originalNumberOfLimbs
    w1_size = n_inputs * n_hidden

    weights = genome["Brain"]
    W1 = weights[:w1_size].reshape(n_inputs, n_hidden)
    W2 = weights[w1_size:].reshape(n_hidden, n_outputs)

    row_start = remove_idx * numberOfInputs
    row_end = row_start + numberOfInputs

    W1 = np.delete(W1, slice(row_start, row_end), axis=0)
    W2 = np.delete(W2, remove_idx, axis=1)

    newBody = [g for i, g in enumerate(genome["Body"]) if i != remove_idx]

    for limb in newBody:
        if limb.parentIndex > remove_idx:
            limb.parentIndex -= 1

    newWeights = np.concatenate([W1.ravel(), W2.ravel()])
    newGenome = {"Body": newBody, "Brain": newWeights}

    return newGenome

def MutateGenome(genome):
    weights = genome["Brain"]
    rng = np.random.default_rng()
    mask = rng.random(weights.shape) < 0.15
    noise = rng.normal(0, 0.2, weights.shape)
    spike = rng.normal(0, 1, weights.shape)
    use_spike = rng.random(weights.shape) < 0.05
    weights += mask * np.where(use_spike, spike, noise)
    weights = np.clip(weights, -10, 10)

    genome["Brain"] = weights

    for i, limb in enumerate(genome["Body"]):
        if random.random() < 0.05:
            genome["Body"][i].parentAnchor *= -1
        if rng.random() < 0.15:
            genome["Body"][i].angleOffset += (rng.normal(0, 0.3))
        if rng.random() < 0.15:
            genome["Body"][i].length = np.clip(
                genome["Body"][i].length + rng.normal(0, 8),
                20, 400
            )
    if random.random() < 0.15:
        newGenome = AddLimbToBrain(genome)
        limb = Limb()
        limb.parentIndex=random.randint(-1, len(genome["Body"]) - 2)
        limb.parentAnchor=1 if random.random() < 0.5 else -1
        newGenome["Body"].append(limb)
        genome = newGenome
    elif random.random() < 0.05:
        newGenome = RemoveOuterLimb(genome)
        genome = newGenome
    elif rng.random() < 0.05:
        parent_indices = {gene.parentIndex for gene in genome["Body"]}
        leaves = [i for i in range(len(genome["Body"])) if i not in parent_indices]
        if leaves:
            target = random.choice(leaves)
            # New parent must have a lower index to keep the tree valid
            new_parent = int(rng.integers(-1, target))
            genome["Body"][target].parentIndex = new_parent

    return genome

def DrawCreatureOverview(genomes, space):
    creatureSize = 220
    creaturesPerRow = np.floor(xScreen / creatureSize)
    sizeModifier = 0.7
    for i, genome in enumerate(genomes):
        xInd = i % creaturesPerRow + 0.5
        yInd = i // creaturesPerRow + 0.5
        torso, height = MakeTorso(space, (xInd * creatureSize - xScreen / 2, yScreen - yInd * creatureSize), sizeModifier)
        BuildBody(space, genome, torso, sizeModifier)

def DrawBestBrain(brain, screen):
    w = 0
    maxWeight = max(1, max(brain.weights))
    minWeight = min(-1, min(brain.weights))
    for input in range(brain.n_inputs):
        pos = (xScreen / 2 - 400, yScreen / 2 + 100 * (input - brain.n_inputs / 2 + 0.5))
        pg.draw.circle(screen, (0, 0, 0), pos, 20)

        for hidden in range(brain.n_hidden):
            posNext = (xScreen / 2, yScreen / 2 + 100 * (hidden - brain.n_hidden / 2 + 0.5))
            weight = brain.weights[w]
            weightNorm = (weight - minWeight) / (maxWeight - minWeight)
            pg.draw.line(screen, (255 * (1 - weightNorm), 255 * weightNorm, 0), pos, posNext)
            w += 1

    for hidden in range(brain.n_hidden):
        pos = (xScreen / 2, yScreen / 2 + 100 * (hidden - brain.n_hidden / 2 + 0.5))
        pg.draw.circle(screen, (0, 0, 0), pos, 20)

        for output in range(brain.n_outputs):
            posNext = (xScreen / 2 + 400, yScreen / 2 + 100 * (output - brain.n_outputs / 2 + 0.5))
            pg.draw.circle(screen, (0, 0, 0), posNext, 20)
            weight = brain.weights[w]
            weightNorm = (weight - minWeight) / (maxWeight - minWeight)
            pg.draw.line(screen, (255 * (1 - weightNorm), 255 * weightNorm, 0), pos, posNext)
            w += 1

def Save(graph, fitnessHistory, epoch):
    graph = plotter.PlotFitnessExport(fitnessHistory, figsize=(6,4), dpi=200)
    if fitnessHistory == []: fitnessHistory = [0]
    graph.get_figure().savefig(f"fitnessHistory_{epoch}_{fitnessHistory[-1]}.png")

def DrawActionSequence(creatures, genomes, space):
    ground_body = pm.Body(body_type=pm.Body.STATIC)
    ground = pm.Segment(ground_body, (-2 * xScreen, 0), (xScreen * 10, 0), 5)
    ground.friction = 1
    space.add(ground_body, ground)

    sortedPairs = sorted(zip(creatures, genomes), key=lambda c: c[0][2].position.x)
    sortedCreatures, sortedGenomes = map(list, zip(*sortedPairs))

    bestCreature = sortedCreatures[-1]
    bestGenome = sortedGenomes[-1]

    torso, height = MakeTorso(space)
    bodies, motors, parentBodies = BuildBody(space, bestGenome, torso)

    bestCreature = [bodies, motors, torso, bestGenome["Brain"], parentBodies]

    return bestCreature

yFloor = 0.8 * yScreen

space = pm.Space()
space.gravity = 0,-981

ground_body = pm.Body(body_type=pm.Body.STATIC)
ground = pm.Segment(ground_body, (-2 * xScreen, 0), (xScreen * 10, 0), 5)
ground.friction = 1
space.add(ground_body, ground)

plotter = Plotter()

genomes = CreateInititalSample(40)
creatures = []
bestBrain = None
bestCreature = None
bestCreatureSpace = None

fitnessHistory = []
graph = plotter.FitnessHistory(fitnessHistory)
screen.blit(graph, (500, 400))

for genome in genomes:
    torso, height = MakeTorso(space)
    bodies, motors, parentBodies = BuildBody(space, genome, torso_body=torso)
    brain = genome["Brain"]
    creatures.append([bodies, motors, torso, brain, parentBodies])

generationNumber = 0
generationTime = 5
t = 0
t_as = 0
dt = 1 / 60

i_as = 0

showCreatures = False
showBrain = False
showBestCreature = False
followBestCreature = False

while running:
    mouse = pg.mouse.get_pos()
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

        if event.type == pg.MOUSEBUTTONDOWN:
            if mouse[0] < xScreen / 10 and mouse[1] < yScreen / 10:
                showCreatures = True
                screen.fill("white")
                creatureOverviewSpace = pm.Space()
                DrawCreatureOverview(genomes, creatureOverviewSpace)
                DrawSpace(screen, creatureOverviewSpace, 0)
            elif xScreen / 10 < mouse[0] < 2 * xScreen / 10 and mouse[1] < yScreen / 10:
                showBrain = True
                screen.fill("white")
                DrawBestBrain(bestBrain, screen)
            elif xScreen / 10 * 2 < mouse[0] < 3 * xScreen / 10 and mouse[1] < yScreen / 10:
                showBestCreature = True
                screen.fill("white")
                bestCreatureSpace = pm.Space()
                bestCreatureSpace.gravity = 0, -981
                bestCreature = DrawActionSequence(creatures, genomes, bestCreatureSpace)
                t_as = 0
                i_as = 0
            elif mouse[0] > xScreen / 10 * 9 and mouse[1] < yScreen / 10:
                Save(graph, fitnessHistory, generationNumber)
            elif mouse[0] > xScreen / 10 * 9 and mouse[1] > yScreen / 10 * 9:
                running = False

    while showCreatures:
        mouse = pg.mouse.get_pos()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

            if event.type == pg.MOUSEBUTTONDOWN:
                if mouse[0] < xScreen / 10 and mouse[1] < yScreen / 10:
                    showCreatures = False

        pg.display.flip()

    while showBrain:
        mouse = pg.mouse.get_pos()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

            if event.type == pg.MOUSEBUTTONDOWN:
                if mouse[0] < xScreen / 10 and mouse[1] < yScreen / 10:
                    showBrain = False
        pg.display.flip()

    while showBestCreature:
        screen.fill("white")
        body = bestCreature[0]
        motors = bestCreature[1]
        torso = bestCreature[2]
        brain = Brain(n_limbs=len(body))
        brain.weights = bestCreature[3]
        parentBodies = bestCreature[4]
        inputs = GetInputs(torso, body, parentBodies)
        out = brain.forward(inputs)
        ApplyOutputs(out, motors)

        for _ in range(10):
            bestCreatureSpace.step(dt / 10)

        mouse = pg.mouse.get_pos()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

            if event.type == pg.MOUSEBUTTONDOWN:
                if mouse[0] < xScreen / 10 and mouse[1] < yScreen / 10:
                    showBestCreature = False
                    bestCreature = None
                elif xScreen / 10 < mouse[0] < xScreen / 10 * 2 and mouse[1] < yScreen / 10:
                    if not followBestCreature:
                        followBestCreature = True
                    else:
                        followBestCreature = False

        DrawSpace(screen, bestCreatureSpace, torso.position.x)

        pg.display.flip()
        clock.tick(60)
        t_as += dt

        if t_as > 0.1 and i_as < 40:
            t_as = 0

            with mss.mss() as sct:
                monitor = {"top": int(yScreen / 10 * 6), "left": int(xScreen / 10 * 3.5), "width": int(xScreen / 10 * 3), "height": int(yScreen / 10 * 4.5)}
                output = f"ActionSequence/ActionSequence_{i_as}.png".format(**monitor)

                sct_img = sct.grab(monitor)

                mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)

            i_as += 1

    screen.fill("white")

    for _ in range(10):
        space.step(dt / 10)

    furthestDistance = 0
    for i, creature in enumerate(creatures):
        body = creature[0]
        motors = creature[1]
        torso = creature[2]
        brain = Brain(n_limbs=len(body))
        brain.weights = creature[3]
        parentBodies = creature[4]
        inputs = GetInputs(torso, body, parentBodies)
        out = brain.forward(inputs)
        ApplyOutputs(out, motors)

        if torso.position.x > furthestDistance:
            furthestDistance = torso.position.x

            if bestBrain is None:
                bestBrain = brain
            if t > generationTime:
                bestBrain = brain

    if generationNumber == 50:
        print(fitnessHistory)
        running = False

    if t > generationTime:
        t = 0
        genomes = EndGeneration(creatures, genomes)
        bodies = []
        motors = []
        creatures = []
        space = pm.Space()
        space.gravity = 0, -981

        ground_body = pm.Body(body_type=pm.Body.STATIC)
        ground = pm.Segment(ground_body, (-2 * xScreen, 0), (xScreen * 10, 0), 5)
        ground.friction = 1
        space.add(ground_body, ground)
        for genome in genomes:
            torso, height = MakeTorso(space)
            bodies, motors, parentBodies = BuildBody(space, genome, torso_body=torso)
            weights = genome["Brain"]
            creatures.append([bodies, motors, torso, weights, parentBodies])

        graph = plotter.FitnessHistory(fitnessHistory)
        screen.blit(graph, (500, 400))
        generationNumber += 1
    t += dt

    DrawGame(screen, furthestDistance)

    DrawSpace(screen, space, furthestDistance)

    screen.blit(graph, (xScreen - plotter.fig_size[0] * plotter.dpi * 1.5, 0))

    pg.display.flip()

    clock.tick(60)

pg.quit()