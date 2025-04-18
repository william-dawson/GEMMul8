program test_dgemm_random
  implicit none
  integer, parameter :: N = 2
  double precision :: A(N,N), B(N,N), C(N,N), D(N,N)
  double precision :: alpha, beta, diff, max_diff, frob_diff
  integer :: i, j, k

  interface
    subroutine dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) bind(C, name="dgemm_")
      implicit none
      character(len=1), intent(in) :: transa, transb
      integer, intent(in) :: m, n, k, lda, ldb, ldc
      double precision, intent(in) :: alpha, beta
      double precision, intent(in) :: A(lda,*), B(ldb,*)
      double precision, intent(inout) :: C(ldc,*)
    end subroutine
  end interface

  call random_seed()
  call random_number(A)
  call random_number(B)

  alpha = 1d0
  beta  = 0d0

  call dgemm('N','N', N, N, N, alpha, A, N, B, N, beta, C, N)

  D = 0d0
  do i = 1, N
    do j = 1, N
      do k = 1, N
        D(i,j) = D(i,j) + A(i,k) * B(k,j)
      end do
    end do
  end do

  max_diff = 0d0
  frob_diff = 0d0
  do i = 1, N
    do j = 1, N
      diff = abs(C(i,j) - D(i,j))
      if (diff > max_diff) max_diff = diff
      frob_diff = frob_diff + diff*diff
    end do
  end do
  frob_diff = sqrt(frob_diff)

  print *, 'Max absolute difference:', max_diff
  print *, 'Frobenius norm of diff:', frob_diff
end program test_dgemm_random
