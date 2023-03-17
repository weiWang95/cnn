package g2048

import (
	"fmt"
	"math/rand"
)

type Area struct {
	w    int
	h    int
	seed int64

	data    [][]uint
	newData [][]uint
	score   int
}

func newArea(w, h int, seed int64) *Area {
	return &Area{
		w:    w,
		h:    h,
		seed: seed,
	}
}

func (a *Area) init() {
	rand.Seed(a.seed)

	a.data = make([][]uint, a.h)
	for i, _ := range a.data {
		a.data[i] = make([]uint, a.w)
	}

	a.newData = make([][]uint, a.h)
	for i, _ := range a.newData {
		a.newData[i] = make([]uint, a.w)
	}
	a.score = 0
}

func (a *Area) random() bool {
	if a.isFull(a.data) {
		return false
	}

	v := 2
	if rand.Intn(2) == 1 {
		v = 4
	}

	for {
		x := rand.Intn(a.w)
		y := rand.Intn(a.h)
		if a.data[y][x] == 0 {
			a.data[y][x] = uint(v)
			break
		}
	}

	return true
}

func (a *Area) isFull(data [][]uint) bool {
	for _, item := range data {
		for _, i := range item {
			if i == 0 {
				return false
			}
		}
	}

	return true
}

func (a *Area) isEnd(data [][]uint) bool {
	for i, _ := range data {
		for j, v := range data[i] {
			if data[i][j] == 0 {
				return false
			}

			if j+1 < a.w && v == data[i][j+1] {
				return false
			}
			if i+1 < a.h && v == data[i+1][j] {
				return false
			}
		}
	}

	return true
}

func (a *Area) max(data [][]uint) uint {
	var max uint
	for i, _ := range data {
		for j, _ := range data[i] {
			if data[i][j] > max {
				max = data[i][j]
			}
		}
	}

	return max
}

func (a *Area) left() int {
	var score int
	a.copyData(a.data, a.newData)
	a.newData, score = a.doLeft(a.newData)
	return score
}

func (a *Area) doLeft(data [][]uint) ([][]uint, int) {
	var score int
	for y, _ := range data {
		position := 0

		for x := 1; x < a.w; x++ {
			if data[y][x] == 0 {
				continue
			}

			if data[y][position] == 0 {
				data[y][position] = data[y][x]
				data[y][x] = 0
				continue
			}

			if data[y][position] == data[y][x] {
				data[y][position] = data[y][position] * 2
				data[y][x] = 0
				score += int(data[y][position])
				position++
			} else {
				position++
				if position != x {
					data[y][position] = data[y][x]
					data[y][x] = 0
				}
			}
		}
	}

	return data, score
}

func (a *Area) right() int {
	var score int
	a.copyData(a.data, a.newData)
	a.newData, score = a.doRight(a.newData)
	return score
}

func (a *Area) doRight(data [][]uint) ([][]uint, int) {
	var score int
	for y, _ := range data {
		position := a.w - 1

		for x := position - 1; x >= 0; x-- {
			if data[y][x] == 0 {
				continue
			}

			if data[y][position] == 0 {
				data[y][position] = data[y][x]
				data[y][x] = 0
				continue
			}

			if data[y][position] == data[y][x] {
				data[y][position] = data[y][position] * 2
				data[y][x] = 0
				score += int(data[y][position])
				position--
			} else {
				position--
				if position != x {
					data[y][position] = data[y][x]
					data[y][x] = 0
				}
			}
		}
	}

	return data, score
}

func (a *Area) up() int {
	var score int
	a.copyData(a.data, a.newData)
	a.newData, score = a.doUp(a.newData)
	return score
}

func (a *Area) doUp(data [][]uint) ([][]uint, int) {
	var score int
	for x := 0; x < a.w; x++ {
		position := 0
		for y := position + 1; y < a.h; y++ {
			if data[y][x] == 0 {
				continue
			}

			if data[position][x] == 0 {
				data[position][x] = data[y][x]
				data[y][x] = 0
				continue
			}

			if data[position][x] == data[y][x] {
				data[position][x] = data[position][x] * 2
				data[y][x] = 0
				score += int(data[position][x])
				position++
			} else {
				position++
				if position != y {
					data[position][x] = data[y][x]
					data[y][x] = 0
				}
			}
		}
	}

	return data, score
}

func (a *Area) down() int {
	var score int
	a.copyData(a.data, a.newData)
	a.newData, score = a.doDown(a.newData)
	return score
}

func (a *Area) doDown(data [][]uint) ([][]uint, int) {
	var score int
	for x := 0; x < a.w; x++ {
		position := a.h - 1
		for y := position - 1; y >= 0; y-- {
			if data[y][x] == 0 {
				continue
			}

			if data[position][x] == 0 {
				data[position][x] = data[y][x]
				data[y][x] = 0
				continue
			}

			if data[position][x] == data[y][x] {
				data[position][x] = data[position][x] * 2
				score += int(data[position][x])
				data[y][x] = 0
				position--
			} else {
				position--
				if position != y {
					data[position][x] = data[y][x]
					data[y][x] = 0
				}
			}
		}
	}

	return data, score
}

func (a *Area) apply() {
	a.copyData(a.newData, a.data)
}

func (a *Area) copyData(source, target [][]uint) {
	for i, _ := range source {
		for j, _ := range source[i] {
			target[i][j] = source[i][j]
		}
	}
}

func (a *Area) computeScore(data [][]uint) int {
	var sum int
	for _, row := range data {
		for _, item := range row {
			sum += int(item)
		}
	}
	return sum
}

func (a *Area) SetScore(sum int) {
	a.score = sum
}

func (a *Area) Score() int {
	return a.score
}

func (a *Area) inspect() {

	for y := 0; y < a.h; y++ {
		for x := 0; x < a.w; x++ {
			fmt.Printf("% 4d ", a.data[y][x])
		}
		fmt.Print("\n\n")
	}
	fmt.Print("\n")
}
