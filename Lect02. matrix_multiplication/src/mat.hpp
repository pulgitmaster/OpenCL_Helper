#include <iomanip> // for std::setprecision()
#include <limits>
template <typename T>
class Mat
{
private:
    int width;
    int height;
    T *data;

public:
    Mat()
        : width(0), height(0), data(nullptr)
    {
    }

    Mat(int width, int height, const T *data)
    {
        this->width = width;
        this->height = height;
        this->data = new T[width * height];
        memcpy(this->data, data, sizeof(T) * width * height);
    }

    Mat(int width, int height)
    {
        this->width = width;
        this->height = height;
        this->data = new T[width * height];
    }

    ~Mat()
    {
        width = 0;
        height = 0;
        if (data != nullptr)
        {
            delete[] data;
        }
    }

    const int GetWidth() const
    {
        return width;
    }

    const int GetHeight() const
    {
        return height;
    }

    const T *GetData() const
    {
        return data;
    }

    T *GetDataAddr(int index)
    {
        return &data[index];
    }

    void SetData(const T *data)
    {
        memcpy(this->data, data, sizeof(T) * width * height);
    }

    void SetData(const T *data, const int size)
    {
        memcpy(this->data, data, sizeof(T) * size);
    }

    void SetDataWithScalar(const T val)
    {
        std::fill(&data[0], &data[width * height], val);
    }

    void SetWidth(int width)
    {
        this->width = width;
    }

    void SetHeight(int height)
    {
        this->height = height;
    }

    T &operator[](int index)
    {
        if (index >= width * height)
        {
            std::cout << "Array index out of bound, exiting\n";
            exit(0);
        }
        return data[index];
    }

    const T &operator[](int index) const
    {
        if (index >= width * height)
        {
            std::cout << "Array index out of bound, exiting\n";
            exit(0);
        }
        return data[index];
    }

    void Print()
    {
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                if (col != width - 1)
                {
                    std::cout << std::setprecision(4) << data[row * width + col] << ", ";
                }
                else
                {
                    std::cout << std::setprecision(4) << data[row * width + col] << "\n";
                }
            }
        }
        std::cout << "\n";
    }
};