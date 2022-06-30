package cnn

import "fmt"

type Kernel struct {
	Step  int
	Size  int
	Value []float64
}

func (k *Kernel) Extract(data []float64) float64 {
	if len(data) != k.Len() {
		panic(fmt.Sprintf("error data size: expect %v, got %v", k.Len(), len(data)))
	}

	// for i, v := range data {
	// 	if i != 0 && i%k.Size == 0 {
	// 		fmt.Println("")
	// 	}
	// 	fmt.Print(" ", v)
	// }
	// fmt.Println("")

	var res float64
	for i := 0; i < len(data); i++ {
		res += data[i] * k.Value[i]
	}

	return res / float64(len(data))
}

func (k *Kernel) Len() int {
	return k.Size * k.Size
}

func (k *Kernel) Half() int {
	return k.Size / 2
}

func (k *Kernel) ExtractAll(data [][]float64) [][]float64 {
	var posX, posY int
	res := make([][]float64, len(data))

	for {
		if posY >= len(data) {
			break
		}

		for {
			if posX >= len(data[0]) {
				break
			}
			if res[posY] == nil {
				res[posY] = make([]float64, len(data[0]))
			}

			samplingData := make([]float64, k.Len())
			for i := 0; i < k.Len(); i++ {
				y, x := posY-k.Half()+i/k.Size, posX-k.Half()+i%k.Size
				// fmt.Printf("Y %v %v -> %v\n", posY, i/k.Size, y)
				// fmt.Printf("X %v %v -> %v\n", posX, i%k.Size, x)
				if y < 0 || y >= len(data) || x < 0 || x >= len(data[0]) {
					samplingData[i] = 0
				} else {
					samplingData[i] = data[y][x]
				}
			}

			// fmt.Printf("Y:%v X:%v\n", posY, posX)
			res[posY][posX] = k.Extract(samplingData)

			posX += k.Step
		}

		// fmt.Println("--------")

		posY += k.Step
		posX = 0
	}

	return res
}
