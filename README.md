
 ## Input Data



 ## 레스토랑 Vectorization
 -  key: search_term( 각 레스토랑의 고유 ID)
 -  variables
  -  rest_category: 레스토랑의 카테고리
  -  rest_menu: 각 레스토랑의 대표메뉴
  -  rest_visit_review: 네이버지도 레스토랑 리뷰 수
  -  rest_blog_review : 블로그 리뷰 수

 ## 유저 Vectorization
-  key: User_id(각 유저의 고유 ID)
-  Review_num: 리뷰수
-  follower: 유저의 팔로워 수
-  following: 유저의 팔로잉 수
-  review_text: myplace에 남긴 리뷰 내용
-  tag: myplace에 남긴 음식점별 태그

## naver_tag_review_크롤링
 - search_term: ID값 : 상세주소 + 사업장명으로 이루어져 있음
 - user_id: 리뷰를 남긴 user id , 추천 순
 - review_text:  리뷰 내용
 - date: 일자
 - revisit: 재방문수


