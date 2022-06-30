package cnn

type IPooling interface {
	GetSize() int
	Pooling(data []float64) float64
	PoolingAll(p IPooling, data [][]float64) [][]float64
}

type BasePooling struct {
	Size int
}

func (b BasePooling) GetSize() int {
	return b.Size
}

func (b BasePooling) PoolingAll(p IPooling, data [][]float64) [][]float64 {
	l := p.GetSize()
	maxY := len(data)/l + 1
	if len(data)%l == 0 {
		maxY = len(data) / l
	}
	maxX := len(data[0])/l + 1
	if len(data[0])%l == 0 {
		maxX = len(data[0]) / l
	}

	r := make([][]float64, maxY)

	for y := 0; y < maxY; y++ {
		for x := 0; x < maxX; x++ {
			if r[y] == nil {
				r[y] = make([]float64, maxX)
			}
			d := make([]float64, l*l)

			for i, _ := range d {
				d[i] = data[y*l+i/l][x*l+i%l]
			}
			r[y][x] = p.Pooling(d)
		}
	}

	return r
}

// 极大池化
var _ IPooling = (*MaxPooling)(nil)

type MaxPooling struct {
	BasePooling
}

func (p *MaxPooling) Pooling(data []float64) float64 {
	var max float64
	for _, v := range data {
		if v > max {
			max = v
		}
	}
	return max
}

// 均值池化
var _ IPooling = (*MeanPooling)(nil)

type MeanPooling struct {
	BasePooling
}

func (p *MeanPooling) Pooling(data []float64) float64 {
	var sum float64
	for _, v := range data {
		sum += v
	}

	return sum / float64(len(data))
}
