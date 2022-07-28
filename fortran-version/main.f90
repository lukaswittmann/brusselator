program Brusselator
    use types, only: dp
    use utils, only: savetxt

    implicit none

    !Parameter
    real(dp), parameter :: t = 100.0, dt = 0.01
    real(dp), parameter :: a = 1.0, b = 3.0, k1 = 1.0, k2 = 1.0, k3 = 1.0, k4 = 1.0, Dx = 0.8, Dy = 0.08
    integer, parameter :: iterations = INT(t/dt), array_size = 250
    
    !Variables
    integer :: beginning, rate, end,  iter, file_iter = 1
    real(dp), dimension(array_size,array_size) :: x, y, temp_x

    character (len=10) :: file_name
    real :: pos1, pos2

    !Starting array inizialization
    x = a
    y = b/a

    !Create random starting inhomogeneities
    do iter = 1, 5
        call random_number(pos1)
        call random_number(pos2)
        x(int(pos1*array_size):int(pos1*array_size+2),int(pos2*array_size):int(pos2*array_size+2)) = a * 1.1
    end do

    print *, "Start of Calculation, Iterations:", iterations
    call system_clock(beginning, rate)

    !Iterationloop
    do iter = 1, iterations

        temp_x = x + (k1 * a - k2 * b * x + k3 * y * x ** 2 - k4 * x) * dt &
            + (Dx * (-4.0 * x &
            + cshift(x,  1, DIM=1) + cshift(x,  -1, DIM=1) &
            + cshift(x,  1, DIM=2) + cshift(x,  -1, DIM=2))) * dt 

        y = y + (k2 * b * x - k3 * y * x ** 2) * dt &
            + (Dy * (-4.0 * y &
            + cshift(y,  1, DIM=1) + cshift(y,  -1, DIM=1) &
            + cshift(y,  1, DIM=2) + cshift(y,  -1, DIM=2))) * dt 

        x = temp_x

        !Write arrays to txt file
        if (mod(iter, 10)==0) then
            write (file_name,'(i0)') file_iter
            print *, file_name, int((iter*100)/(iterations)),"%"
            call savetxt("export_x/"//trim(file_name)//".txt", x)
            !call savetxt("export_y/"//trim(file_name)//".txt", y)
            file_iter = file_iter + 1
        end if
    end do

    !Benchmark
    call system_clock(end)
    print *, "Elapsed time: ", real(end - beginning) / real(rate), "seconds."

end program Brusselator

