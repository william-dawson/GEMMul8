subroutine dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,&
        c, ldc)
#ifndef DONTWRAPGEMM
      use iso_c_binding
      implicit none
      interface
      subroutine offload_dgemm(oLayout, oTransA, oTransB, oM, oN, oK, &
                               oAlpha, oA, oLda, oB, oLdb, oBeta,     &
                               oC, oLdc) bind(c)
      import c_int, c_double
      integer(kind=c_int), intent(in), value :: oLayout
      integer(kind=c_int), intent(in), value :: oTransA, oTransB
      integer(kind=c_int), intent(in), value :: oM, oN, oK
      integer(kind=c_int), intent(in), value :: oLda, oLdb, oLdc
      real(kind=c_double), intent(in), value :: oAlpha, oBeta
      real(kind=c_double), intent(in)    :: oA(oLda,*), oB(oLdb,*)
      real(kind=c_double), intent(inout) :: oC(oLdc,*)
      end subroutine offload_dgemm
      end interface
#else
      implicit none
#endif
      character :: transa, transb
      integer(kind=4) :: m, n, k, lda, ldb, ldc
      integer(kind=4) :: lda0, ldb0, lay, ta, tb
      double precision :: alpha, beta
      double precision, dimension(lda,*) :: a
      double precision, dimension(ldb,*) :: b
      double precision, dimension(ldc,*) :: c
      integer(kind=4) :: ka, kb, istat
      character(len=255) :: errmsg
      double precision, dimension(:,:), allocatable :: pA, pB, pC

#ifndef DONTWRAPGEMM
!     .. External Functions ..
      LOGICAL LSAME
      EXTERNAL lsame
      write(6,*) "DGEMM with ", transa, transb, m, n, k, lda, ldb, ldc
      !call flush(6)
      lay = 0
      ta = 0
      tb = 0
      ! early return if possible
      if (m .le. 0 .or. n .le. 0) then
          return
      else if (k .le. 0) then
          c(1:m,1:n) = beta*c(1:m,1:n)
          return
      end if
      ! cuBLAS API Reference guide: For maximum compatibility with
      ! existing Fortran [...], the cuBLAS library uses column-major
      ! -> we are in F hence ColMajor, no need to revert back
      if (lsame(transa,'T')) then
          ta = 1
      else if (lsame(transa,'C')) then
          ta = 2
      end if
      if (lsame(transb,'T')) then
          tb = 1
      else if (lsame(transb,'C')) then
          tb = 2
      end if
#ifndef WRAPWITHPADDING
      call offload_dgemm(lay, ta, tb, m, n, k, alpha, a, lda, b, ldb, &
                         beta, c, ldc)
#else
      if (m .ge. 1024 .and. n .ge. 1024 .and. k .ge. 1024) then
          call offload_dgemm(lay, ta, tb, m, n, k,                    &
                             alpha, a, lda, b, ldb, beta, c, ldc)
          !write(6,*) "JJ", c(1:min(m,2),1:min(n,2)), "...", c(max(ldc-1,1):ldc,max(n-1,1):n)
          return
      end if
      ! padding code to work around a current issue
      lda0 = m
      ka = k
      ldb0 = k
      kb = n
      if (ta .ne. 0) then
          lda0 = k
          ka = m
      end if
      if (tb .ne. 0) then
          ldb0 = n
          kb = k
      end if
      allocate(pA(1:max(1024,lda),1:max(1024,ka)),                    &
               pB(1:max(1024,ldb),1:max(1024,kb)),                    &
               pC(1:max(1024,ldc),1:max(1024,n)),                     &
               STAT=istat, ERRMSG=errmsg)
      if (istat .ne. 0) then
          write(*,*) errmsg, " : istat =", istat
          call abort
      end if
      pA = 0
      pB = 0
      pC = 0
      pA(1:lda0,1:ka) = a(1:lda0,1:ka) ! copy only matrix, not lda part
      pB(1:ldb0,1:kb) = b(1:ldb0,1:kb) ! since it likely contains trash
      pC(1:m,1:n) = c(1:m,1:n)
      call offload_dgemm(lay, ta, tb,                                 &
                         max(1024,m), max(1024,n), max(1024,k),       &
                         alpha,                                       &
                         pA, max(1024,lda),                           &
                         pB, max(1024,ldb),                           &
                         beta,                                        &
                         pC, max(1024,ldc))
      c(1:m,1:n) = pC(1:m,1:n)
      !write(6,*) "JJ", c(1:min(m,2),1:min(n,2)), "...", c(max(ldc-1,1):ldc,max(n-1,1):n)
      deallocate(pA, pB, pC, STAT=istat, ERRMSG=errmsg)
      if (istat .ne. 0) then
          write(*,*) errmsg, " : istat =", istat
          call abort
      end if
#endif
#else
!https://netlib.org/lapack/explore-html/d7/d2b/dgemm_8f_source.html
!     .. External Functions ..
      LOGICAL LSAME
      EXTERNAL lsame
!     ..
!     .. External Subroutines ..
      EXTERNAL xerbla
!     ..
!     .. Intrinsic Functions ..
      INTRINSIC max
!     ..
!     .. Local Scalars ..
      DOUBLE PRECISION TEMP
      INTEGER I,INFO,J,L,NROWA,NROWB
      LOGICAL NOTA,NOTB
!     ..
!     .. Parameters ..
      DOUBLE PRECISION ONE,ZERO
      parameter(one=1.0d+0,zero=0.0d+0)
!     ..
!
!     Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not
!     transposed and set  NROWA and NROWB  as the number of rows of  A
!     and  B  respectively.
!
      !write(6,*) "DGEMM with ", transa, transb, m, n, k, lda, ldb, ldc
      !call flush(6)
      nota = lsame(transa,'N')
      notb = lsame(transb,'N')
      IF (nota) THEN
          nrowa = m
      ELSE
          nrowa = k
      END IF
      IF (notb) THEN
          nrowb = k
      ELSE
          nrowb = n
      END IF
!
!     Test the input parameters.
!
      info = 0
      IF ((.NOT.nota) .AND. (.NOT.lsame(transa,'C')) .AND.            &
          (.NOT.lsame(transa,'T'))) THEN
          info = 1
      ELSE IF ((.NOT.notb) .AND. (.NOT.lsame(transb,'C')) .AND.       &
               (.NOT.lsame(transb,'T'))) THEN
          info = 2
      ELSE IF (m.LT.0) THEN
          info = 3
      ELSE IF (n.LT.0) THEN
          info = 4
      ELSE IF (k.LT.0) THEN
          info = 5
      ELSE IF (lda.LT.max(1,nrowa)) THEN
          info = 8
      ELSE IF (ldb.LT.max(1,nrowb)) THEN
          info = 10
      ELSE IF (ldc.LT.max(1,m)) THEN
          info = 13
      END IF
      IF (info.NE.0) THEN
          CALL xerbla('DGEMM ',info)
          RETURN
      END IF
!
!     Quick return if possible.
!
      IF ((m.EQ.0) .OR. (n.EQ.0) .OR.                                 &
          (((alpha.EQ.zero).OR. (k.EQ.0)).AND. (beta.EQ.one))) RETURN
!
!     And if  alpha.eq.zero.
!
      IF (alpha.EQ.zero) THEN
          IF (beta.EQ.zero) THEN
              DO 20 j = 1,n
                  DO 10 i = 1,m
                      c(i,j) = zero
   10             CONTINUE
   20         CONTINUE
          ELSE
              DO 40 j = 1,n
                  DO 30 i = 1,m
                      c(i,j) = beta*c(i,j)
   30             CONTINUE
   40         CONTINUE
          END IF
          RETURN
      END IF
!
!     Start the operations.
!
      IF (notb) THEN
          IF (nota) THEN
!
!           Form  C := alpha*A*B + beta*C.
!
              DO 90 j = 1,n
                  IF (beta.EQ.zero) THEN
                      DO 50 i = 1,m
                          c(i,j) = zero
   50                 CONTINUE
                  ELSE IF (beta.NE.one) THEN
                      DO 60 i = 1,m
                          c(i,j) = beta*c(i,j)
   60                 CONTINUE
                  END IF
                  DO 80 l = 1,k
                      temp = alpha*b(l,j)
                      DO 70 i = 1,m
                          c(i,j) = c(i,j) + temp*a(i,l)
   70                 CONTINUE
   80             CONTINUE
   90         CONTINUE
          ELSE
!
!           Form  C := alpha*A**T*B + beta*C
!
              DO 120 j = 1,n
                  DO 110 i = 1,m
                      temp = zero
                      DO 100 l = 1,k
                          temp = temp + a(l,i)*b(l,j)
  100                 CONTINUE
                      IF (beta.EQ.zero) THEN
                          c(i,j) = alpha*temp
                      ELSE
                          c(i,j) = alpha*temp + beta*c(i,j)
                      END IF
  110             CONTINUE
  120         CONTINUE
          END IF
      ELSE
          IF (nota) THEN
!
!           Form  C := alpha*A*B**T + beta*C
!
              DO 170 j = 1,n
                  IF (beta.EQ.zero) THEN
                      DO 130 i = 1,m
                          c(i,j) = zero
  130                 CONTINUE
                  ELSE IF (beta.NE.one) THEN
                      DO 140 i = 1,m
                          c(i,j) = beta*c(i,j)
  140                 CONTINUE
                  END IF
                  DO 160 l = 1,k
                      temp = alpha*b(j,l)
                      DO 150 i = 1,m
                          c(i,j) = c(i,j) + temp*a(i,l)
  150                 CONTINUE
  160             CONTINUE
  170         CONTINUE
          ELSE
!
!           Form  C := alpha*A**T*B**T + beta*C
!
              DO 200 j = 1,n
                  DO 190 i = 1,m
                      temp = zero
                      DO 180 l = 1,k
                          temp = temp + a(l,i)*b(j,l)
  180                 CONTINUE
                      IF (beta.EQ.zero) THEN
                          c(i,j) = alpha*temp
                      ELSE
                          c(i,j) = alpha*temp + beta*c(i,j)
                      END IF
  190             CONTINUE
  200         CONTINUE
          END IF
      END IF
!
#endif
end subroutine

! wrap zgemm by having it just call dgemm under the hood
subroutine zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
  ! Subroutine with the same API as ZGEMM, implemented using DGEMM.
  character(len=1), intent(in) :: transa, transb
  integer, intent(in) :: m, n, k, lda, ldb, ldc
  complex*16, intent(in) :: alpha
  complex*16, intent(in) :: a(lda, k), b(ldb, n)
  complex*16, intent(inout) :: c(ldc, n)
  complex*16, intent(in) :: beta

  real*8, dimension(:,:), allocatable :: ar, ai, br, bi, cr, ci
  integer :: i, j
  integer :: ka, kb
  character(len=1) :: tra, trb

  if (transa == 'N' .or. transa == 'n') then
    ka = k
  else
    ka = m
  endif
  if (transb == 'N' .or. transb == 'n') then
    kb = n
  else
    kb = k
  endif

  tra = transa
  trb = transb
  if (tra == 'C' .or. tra == 'c') then
    tra = 'T'
  end if
  if (trb == 'C' .or. trb == 'c') then
    trb = 'T'
  end if

  allocate(ar(lda, k), ai(lda, k))
  allocate(br(ldb, n), bi(ldb, n))
  allocate(cr(ldc, n), ci(ldc, n))

  ! split real / complex
  do i = 1, lda
    do j = 1, ka
      ar(i, j) = dble(a(i, j))
      ai(i, j) = dimag(a(i, j))
      if (transa == "C" .or. transa == "c") ai(i, j) = -ai(i, j)
    end do
  end do

  do i = 1, ldb
    do j = 1, kb
      br(i, j) = dble(b(i, j))
      bi(i, j) = dimag(b(i, j))
      if (transb == "C" .or. transb == "c") bi(i, j) = -bi(i, j)
    end do
  end do

  cr = 0
  ci = 0

  ! multiply parts
  call dgemm(tra, trb, m, n, k, dble(alpha), ar, lda, br, ldb, dble(1), cr, ldc)
  call dgemm(tra, trb, m, n, k, -dble(alpha), ai, lda, bi, ldb, dble(1), cr, ldc)
  call dgemm(tra, trb, m, n, k, dble(alpha), ar, lda, bi, ldb, dble(1), ci, ldc)
  call dgemm(tra, trb, m, n, k, dble(alpha), ai, lda, br, ldb, dble(1), ci, ldc)

  ! accumulate
  do i = 1, ldc
    do j = 1, n
      c(i, j) = beta * c(i, j) + dcmplx(cr(i, j), ci(i, j))
    end do
  end do

  deallocate(ar, ai, br, bi, cr, ci)
end subroutine zgemm

