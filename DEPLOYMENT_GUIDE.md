# GitHub Pages 배포 가이드
## 백테스트 대시보드를 인터넷에서 보는 법 (완전 초보자용)

> 이 가이드를 따라하면 **어디서든 브라우저로 볼 수 있는 웹페이지**가 만들어지고,
> **매일 오전 9시(한국시간)** 에 자동으로 업데이트됩니다.
> GitHub/터미널 경험이 전혀 없어도 OK입니다. 한 단계씩 따라오세요.

---

## 목차

1. [사전 준비 — GitHub 계정 만들기](#1-사전-준비--github-계정-만들기)
2. [Git 설치하기 (Mac)](#2-git-설치하기-mac)
3. [binance_bot-5.py 파일 레포에 넣기](#3-binance_bot-5py-파일-레포에-넣기)
4. [GitHub에 새 레포지토리 만들기](#4-github에-새-레포지토리-만들기)
5. [터미널로 파일 올리기 (git push)](#5-터미널로-파일-올리기-git-push)
6. [GitHub Pages 활성화](#6-github-pages-활성화)
7. [GitHub Actions 권한 설정](#7-github-actions-권한-설정)
8. [첫 번째 리포트 수동 실행](#8-첫-번째-리포트-수동-실행)
9. [웹페이지 확인 & URL](#9-웹페이지-확인--url)
10. [매일 자동 업데이트 확인 방법](#10-매일-자동-업데이트-확인-방법)
11. [트러블슈팅](#11-트러블슈팅)

---

## 1. 사전 준비 — GitHub 계정 만들기

> 이미 GitHub 계정이 있으면 이 단계를 건너뛰세요.

1. 브라우저에서 **https://github.com** 에 접속합니다.
2. 오른쪽 위의 **Sign up** 버튼을 클릭합니다.
3. 이메일, 비밀번호, 사용자 이름(영어)을 입력하고 계정을 만듭니다.
4. 이메일 인증을 완료합니다.

---

## 2. Git 설치하기 (Mac)

터미널(Terminal)을 열고 아래 명령어를 입력하세요.
터미널은 **Spotlight(⌘+Space) → "Terminal" 검색** 으로 열 수 있습니다.

```bash
git --version
```

결과로 `git version 2.x.x` 같은 것이 나오면 이미 설치되어 있습니다. → **3단계로 이동**

설치가 안 되어 있으면 아래 명령어를 입력하세요:

```bash
xcode-select --install
```

팝업이 뜨면 **Install** 을 클릭하고 완료될 때까지 기다립니다 (수 분 소요).

---

## 3. binance_bot-5.py 파일 레포에 넣기

`generate_report.py`가 실행될 때 **같은 폴더**에 `binance_bot-5.py` 파일이 있어야 합니다.
이 파일은 Binance 봇의 설정(어떤 코인을 어떤 파라미터로 거래하는지)을 담고 있습니다.

현재 `binance_bot-5.py` 는 `~/Downloads/` 에 있습니다.
아래 명령어로 프로젝트 폴더에 복사하세요:

```bash
cp ~/Downloads/binance_bot-5.py \
   ~/Downloads/260208\ 자동매매전략\ 백테스트/binance_bot-5.py
```

> **주의**: `binance_bot-5.py` 에 API 키나 시크릿이 있으면 레포에 올리기 전에
> 해당 내용을 지우거나 별도 파일로 분리하세요.
> 이 파일은 설정값(파라미터)만 읽고 실제 거래 실행은 하지 않습니다.

---

## 4. GitHub에 새 레포지토리 만들기

1. **https://github.com** 에 로그인합니다.
2. 오른쪽 위 **+** 버튼 → **New repository** 클릭.
3. 아래와 같이 설정합니다:

   | 항목 | 값 |
   |------|---|
   | Repository name | `backtest-dashboard` (원하는 이름으로) |
   | Description | `4H 백테스트 자동 대시보드` |
   | Public / Private | **Public** (GitHub Pages 무료 사용을 위해) |
   | Initialize... | **체크 안 함** (로컬 파일을 올릴 것이므로) |

4. **Create repository** 버튼 클릭.

5. 생성된 페이지에서 레포 URL을 복사해 둡니다.
   형식: `https://github.com/사용자이름/backtest-dashboard.git`

---

## 5. 터미널로 파일 올리기 (git push)

터미널을 열고 아래 명령어를 **한 줄씩** 입력하세요.

### 5-1. 프로젝트 폴더로 이동

```bash
cd ~/Downloads/260208\ 자동매매전략\ 백테스트
```

> `cd` 는 폴더 이동 명령어입니다. `\` 는 공백을 가진 폴더명을 처리하기 위한 것입니다.

### 5-2. Git 초기화

```bash
git init
```

> 이 폴더를 Git이 관리하도록 초기화합니다. `.git` 이라는 숨김 폴더가 생깁니다.

### 5-3. 내 이름과 이메일 설정 (GitHub 계정과 동일하게)

```bash
git config user.name "홍길동"
git config user.email "your@email.com"
```

> 따옴표 안에 본인의 이름과 GitHub 가입 이메일을 넣으세요.

### 5-4. 파일 전체를 스테이징

```bash
git add .
```

> `.` 은 현재 폴더의 모든 파일을 뜻합니다. (`.gitignore`에 적힌 파일은 제외됩니다)

### 5-5. 첫 번째 커밋

```bash
git commit -m "첫 번째 업로드"
```

> 저장 이유를 메모로 남기는 것입니다.

### 5-6. GitHub 레포와 연결

```bash
git remote add origin https://github.com/사용자이름/backtest-dashboard.git
```

> **주의**: `사용자이름/backtest-dashboard` 부분을 4단계에서 만든 본인 레포 URL로 바꾸세요.

### 5-7. 업로드 (push)

```bash
git branch -M main
git push -u origin main
```

> 처음 push 할 때 GitHub 로그인 창이 뜰 수 있습니다.
> 팝업이 뜨면 **GitHub 계정으로 로그인**하거나,
> Personal Access Token을 입력합니다.

#### 비밀번호 대신 Personal Access Token(PAT) 사용하는 법

GitHub는 2021년부터 비밀번호 대신 토큰을 사용합니다.

1. GitHub → 오른쪽 위 프로필 사진 → **Settings**
2. 왼쪽 메뉴 맨 아래 → **Developer settings**
3. **Personal access tokens** → **Tokens (classic)** → **Generate new token**
4. Note: `backtest-push`, Expiration: `No expiration` (또는 원하는 기간)
5. **repo** 체크박스 선택 → **Generate token**
6. 생성된 토큰(`ghp_...`)을 복사해 저장해 두세요 (다시 볼 수 없음)
7. `git push` 시 비밀번호 입력란에 이 토큰을 붙여넣습니다.

---

## 6. GitHub Pages 활성화

1. GitHub에서 본인의 레포(`backtest-dashboard`) 페이지로 이동합니다.
2. 상단 탭에서 **Settings** 클릭.
3. 왼쪽 사이드바에서 **Pages** 클릭 (Code and automation 섹션).
4. **Source** 를 **Deploy from a branch** 로 선택.
5. **Branch** 드롭다운에서 `main` 선택, 폴더는 `/docs` 선택.
6. **Save** 버튼 클릭.

잠시 후 (1~3분) 페이지 상단에 아래와 같은 메시지가 표시됩니다:

```
Your site is live at https://사용자이름.github.io/backtest-dashboard/
```

> 이 URL이 여러분의 백테스트 대시보드 주소입니다!

---

## 7. GitHub Actions 권한 설정

GitHub Actions가 백테스트 결과를 레포에 자동으로 커밋하려면 쓰기 권한이 필요합니다.

1. 레포 **Settings** → **Actions** → **General** 클릭.
2. 페이지 아래쪽 **Workflow permissions** 섹션을 찾습니다.
3. **Read and write permissions** 를 선택합니다.
4. **Save** 버튼 클릭.

---

## 8. 첫 번째 리포트 수동 실행

GitHub Actions는 매일 자동으로 실행되지만, 지금 당장 결과를 보고 싶다면:

1. 레포 상단 탭 **Actions** 클릭.
2. 왼쪽 목록에서 **Daily Backtest Report** 클릭.
3. 오른쪽 위 **Run workflow** 버튼 클릭 → **Run workflow** 확인.
4. 실행 중인 워크플로우가 나타납니다. 클릭하면 진행 상황을 볼 수 있습니다.
5. 완료되면 (보통 20~40분 소요) ✅ 녹색 체크가 표시됩니다.

실행이 완료되면 `docs/charts/` 에 차트 이미지가, `docs/data/metrics.json` 에
지표 데이터가 자동으로 추가되고 웹페이지가 업데이트됩니다.

---

## 9. 웹페이지 확인 & URL

대시보드 URL 형식:
```
https://사용자이름.github.io/backtest-dashboard/
```

예시: GitHub 아이디가 `trader123` 이고 레포 이름이 `backtest-dashboard` 라면:
```
https://trader123.github.io/backtest-dashboard/
```

> GitHub Pages 반영까지 최대 5분이 걸릴 수 있습니다.
> 브라우저 캐시 때문에 안 보이면 **Shift+새로고침(⇧+⌘+R)** 을 해보세요.

---

## 10. 매일 자동 업데이트 확인 방법

매일 KST 09:00 이후에 아래를 확인하세요:

1. 레포 **Actions** 탭에서 오늘 날짜의 워크플로우 실행 결과 확인.
2. ✅ 초록색이면 성공, ❌ 빨간색이면 실패.
3. 실패 시 해당 항목을 클릭해 오류 로그를 확인할 수 있습니다.

웹페이지 상단에도 **마지막 업데이트** 시간이 표시됩니다.

---

## 11. 트러블슈팅

### ❌ "binance_bot-5.py 파일을 찾을 수 없습니다" 오류

**원인**: `binance_bot-5.py` 가 레포에 없음.

**해결**:
```bash
cp ~/Downloads/binance_bot-5.py \
   ~/Downloads/260208\ 자동매매전략\ 백테스트/binance_bot-5.py
cd ~/Downloads/260208\ 자동매매전략\ 백테스트
git add binance_bot-5.py
git commit -m "Add binance_bot-5.py"
git push
```

---

### ❌ GitHub Pages 에서 차트가 보이지 않음 (빈 화면)

**원인**: 첫 번째 Actions 실행 전이라 `docs/charts/*.png` 가 없음.

**해결**: 8단계 "첫 번째 리포트 수동 실행" 을 진행하세요.

---

### ❌ Actions 탭에서 "permission denied" 오류

**원인**: 7단계 Actions 쓰기 권한 설정이 안 됨.

**해결**: Settings → Actions → General → Workflow permissions → Read and write 선택.

---

### ❌ git push 할 때 "Authentication failed"

**원인**: 비밀번호 대신 토큰을 사용해야 함.

**해결**: 5단계의 "Personal Access Token 사용하는 법" 참고.

---

### ❌ Actions 실행 시간이 너무 오래 걸림 (60분 초과)

**원인**: Binance API에서 데이터를 처음 다운받는 데 오래 걸림.

**해결**: 두 번째 실행부터는 캐시가 적용되어 훨씬 빠릅니다 (보통 15~25분).

---

### ❌ 코드를 수정하고 반영하고 싶을 때

```bash
cd ~/Downloads/260208\ 자동매매전략\ 백테스트
git add .
git commit -m "코드 수정 내용 메모"
git push
```

---

## 최종 정리

| 단계 | 해야 할 일 |
|------|-----------|
| 1 | GitHub 계정 만들기 |
| 2 | Git 설치 확인 (`git --version`) |
| 3 | `binance_bot-5.py` 를 프로젝트 폴더에 복사 |
| 4 | GitHub에서 새 레포 생성 (Public) |
| 5 | 터미널에서 `git init` → `git add .` → `git commit` → `git push` |
| 6 | Settings → Pages → `main` 브랜치 `/docs` 폴더 선택 → Save |
| 7 | Settings → Actions → General → Read and write permissions |
| 8 | Actions 탭에서 수동 실행 → 완료 대기 |
| 9 | `https://사용자이름.github.io/레포이름/` 에서 확인 |

**완료 후 매일 KST 09:00에 자동으로 업데이트됩니다!** 🎉
