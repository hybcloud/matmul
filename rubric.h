#define RUBRIC_CPU 0
#define RUBRIC_GPU 1
#define RUBRIC_MPI 2
#define RUBRIC_LENGTH 8

float rubric[3][RUBRIC_LENGTH];

int rubricInit()
{
    //CPU MARKS
    rubric[RUBRIC_CPU][7] = 2;
    rubric[RUBRIC_CPU][6] = 4;
    rubric[RUBRIC_CPU][5] = 8;
    rubric[RUBRIC_CPU][4] = 16;
    rubric[RUBRIC_CPU][3] = 32;
    rubric[RUBRIC_CPU][2] = 64;
    // wrong answer, or doesn't compile
    rubric[RUBRIC_CPU][1] = 1000;
    rubric[RUBRIC_CPU][0] = 1000;

    //GPU MARKS
    rubric[RUBRIC_GPU][7] = 4;
    rubric[RUBRIC_GPU][6] = 8;
    rubric[RUBRIC_GPU][5] = 16;
    rubric[RUBRIC_GPU][4] = 32;
    rubric[RUBRIC_GPU][3] = 64;
    rubric[RUBRIC_GPU][2] = 128;
    // wrong answer, or doesn't compile
    rubric[RUBRIC_GPU][1] = 1000;
    rubric[RUBRIC_GPU][0] = 1000;

    //MPI MARKS
    rubric[RUBRIC_MPI][7] = 1.5;
    rubric[RUBRIC_MPI][6] = 3;
    rubric[RUBRIC_MPI][5] = 6;
    rubric[RUBRIC_MPI][4] = 12;
    rubric[RUBRIC_MPI][3] = 24;
    rubric[RUBRIC_MPI][2] = 48;
    // wrong answer, or doesn't compile
    rubric[RUBRIC_MPI][1] = 1000;
    rubric[RUBRIC_MPI][0] = 1000;
    return 1;
}

float getGrade(float performanceFactor, double err, float* gradeTable)
{
    //floating point error tolerance of the answer given
    const double errTolerance = 1e-17;
    const float fullSpeed = gradeTable[RUBRIC_LENGTH - 1];

    if (abs(err) <= errTolerance)
    {
        // Matrix multiplication works, but is about as slow as possible.
        if (performanceFactor >= gradeTable[2])
        {
            return 2;
        }
        else
        {
            // God-like performance. Full marks.
            if (performanceFactor < gradeTable[RUBRIC_LENGTH - 1])
            {
                return RUBRIC_LENGTH - 1;
            }
            else
            {
                for (int gradeIdx = RUBRIC_LENGTH; gradeIdx >= 1; gradeIdx--)
                {
                    //Apply the logarithmic grade
                    if (performanceFactor > gradeTable[gradeIdx] && performanceFactor <= gradeTable[gradeIdx - 1])
                    {
                        float grade = 7 + log2(fullSpeed) - log2(performanceFactor);
                        return grade;
                    }
                }
            }
        }
    }
    else
    {
        // Matrix multiplication doesn't work properly. 1 point for submitting something that runs at least.
        return 1;
    }
    return 1;
}